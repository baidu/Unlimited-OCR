"""
Concurrent inference via SGLang.

Two input modes are supported:
  1. Dataset images: pass --image_dir and each image is sent as one request.
  2. PDF pages: pass --pdf and each converted page is sent as one request.

Long documents are processed incrementally: each finished page is written
atomically to --output_dir, so an interrupted or failed run can be resumed by
re-running the same command. Pages whose output already exists are skipped
unless --overwrite is given. A run summary (per page status, tokens, timing)
is written to <output_dir>/manifest.json.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SERVED_MODEL_NAME = "Unlimited-OCR"
SERVER_URL = "http://127.0.0.1:10000"
HOST = "0.0.0.0"
PORT = 10000
SERVER_TIMEOUT = 300
PDF_DPI = 300
ATTENTION_BACKEND = "fa3"
PAGE_SIZE = 1
MEM_FRACTION_STATIC = 0.8
PROMPT = "document parsing."
TEMPERATURE = 0
CONTEXT_LENGTH = 32768
NO_REPEAT_NGRAM_SIZE = 35
NGRAM_WINDOW = 128
REQUEST_TIMEOUT = 1200
MAX_RETRIES = 5
NO_REPEAT_NGRAM_PROCESSOR_STR = None
MANIFEST_NAME = "manifest.json"


def get_ngram_processor_str():
    global NO_REPEAT_NGRAM_PROCESSOR_STR
    if NO_REPEAT_NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        NO_REPEAT_NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return NO_REPEAT_NGRAM_PROCESSOR_STR


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 300) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc):
        out_path = os.path.join(out_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out_path)
        image_paths.append(out_path)
    doc.close()
    return image_paths


def encode_image(image_path: str) -> dict:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(image_path: str) -> list[dict]:
    return [{"type": "text", "text": PROMPT}, encode_image(image_path)]


def is_completed(output_file: str) -> bool:
    """A page counts as done only if its output file exists and is non-empty."""
    return os.path.exists(output_file) and os.path.getsize(output_file) > 0


def remove_partial(output_file: str) -> None:
    try:
        os.remove(f"{output_file}.part")
    except OSError:
        pass


def server_ready(server_url: str) -> bool:
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def start_server(args):
    if server_ready(SERVER_URL):
        print(f"Reuse existing SGLang server: {SERVER_URL}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(args.server_log)) or ".", exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        args.model_dir,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--attention-backend",
        ATTENTION_BACKEND,
        "--page-size",
        str(PAGE_SIZE),
        "--mem-fraction-static",
        str(MEM_FRACTION_STATIC),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    print(f"Starting SGLang server on GPU {args.gpu}, port {PORT} ...")
    log_file = open(args.server_log, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    process._log_file = log_file
    print(f"Server PID: {process.pid}")

    start = time.time()
    while time.time() - start < SERVER_TIMEOUT:
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(f"SGLang server exited early. Check {args.server_log}")
        if server_ready(SERVER_URL):
            print(f"Server ready ({time.time() - start:.0f}s)")
            return process
        time.sleep(3)

    stop_server(process)
    raise TimeoutError(f"Timed out waiting for SGLang server. Check {args.server_log}")


def stop_server(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    process._log_file.close()


def collect_stream_silent(resp, output_file: str | None) -> dict:
    """Stream the response. The output file is written atomically: deltas go to
    a ``.part`` sibling and it is renamed into place only after the full stream
    is consumed, so a present output file always means a complete page."""
    chunks = []
    token_count = 0
    first_token_time = None
    tmp_path = f"{output_file}.part" if output_file else None
    f = open(tmp_path, "w", encoding="utf-8") if tmp_path else None
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError):
                continue
            if not delta:
                continue
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
            chunks.append(delta)
            if f:
                f.write(delta)
    finally:
        if f:
            f.close()

    # Reached only when the stream completed without raising; promote the
    # partial file to its final name atomically.
    if tmp_path:
        os.replace(tmp_path, output_file)

    end_time = time.time()
    decode_time = (end_time - first_token_time) if first_token_time and token_count > 1 else 0
    return {"tokens": token_count, "decode_time": decode_time, "text": "".join(chunks)}


def infer_one(image_path: str, output_file: str | None, args, idx: int) -> dict:
    name = os.path.basename(image_path)
    base = {"image": image_path, "output": output_file}

    if output_file and not args.overwrite and is_completed(output_file):
        print(f"  [{idx}] {name}: skipped (already parsed)")
        return {**base, "tokens": 0, "decode_time": 0, "text": "", "status": "skipped", "attempts": 0}

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": build_content(image_path)}],
        "temperature": TEMPERATURE,
        "skip_special_tokens": False,
        "stream": True,
        "images_config": {"image_mode": args.image_mode},
    }
    if NO_REPEAT_NGRAM_SIZE > 0 and NGRAM_WINDOW > 0:
        payload["custom_logit_processor"] = get_ngram_processor_str()
        payload["custom_params"] = {
            "ngram_size": NO_REPEAT_NGRAM_SIZE,
            "window_size": NGRAM_WINDOW,
        }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            if resp.status_code == 502 and attempt < MAX_RETRIES - 1:
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            result = collect_stream_silent(resp, output_file)
            print(f"  [{idx}] {name}: {result['tokens']} tokens, {result['decode_time']:.1f}s")
            return {**base, **result, "status": "ok", "attempts": attempt + 1}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [{idx}] {name}: retry {attempt + 1}/{MAX_RETRIES} ({e})")
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [{idx}] {name}: FAILED ({e})")
            if output_file:
                remove_partial(output_file)
            return {**base, "tokens": 0, "decode_time": 0, "text": "", "status": "failed", "attempts": attempt + 1}


def collect_dataset_images(image_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    return sorted(image_files, key=lambda f: os.path.getsize(f), reverse=True)


def build_jobs(args) -> tuple[list[tuple[str, str | None]], str | None]:
    """Return (jobs, tmp_dir). tmp_dir is the temporary directory holding
    rendered PDF pages (to be cleaned up by the caller), or None otherwise."""
    if args.pdf:
        tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
        image_files = pdf_to_images(args.pdf, tmp_dir, dpi=PDF_DPI)
        prefix = os.path.splitext(os.path.basename(args.pdf))[0]
        jobs = []
        for i, image_path in enumerate(image_files):
            output_file = None
            if args.output_dir:
                output_file = os.path.join(args.output_dir, f"{prefix}_page_{i + 1:04d}.md")
            jobs.append((image_path, output_file))
        return jobs, tmp_dir

    if not args.image_dir:
        raise ValueError("Either --image_dir or --pdf is required")
    image_files = collect_dataset_images(args.image_dir)

    jobs = []
    for image_path in image_files:
        output_file = None
        if args.output_dir:
            rel = os.path.relpath(image_path, args.image_dir)
            stem = os.path.splitext(rel)[0].replace(os.sep, "__")
            output_file = os.path.join(args.output_dir, f"{stem}.md")
        jobs.append((image_path, output_file))
    return jobs, None


def write_manifest(args, results: list[dict], wall_time: float) -> str:
    total_tokens = sum(r["tokens"] for r in results)
    manifest = {
        "model": SERVED_MODEL_NAME,
        "image_mode": args.image_mode,
        "wall_time_sec": round(wall_time, 2),
        "total_tokens": total_tokens,
        "counts": {
            "ran": len(results),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
        },
        "jobs": [
            {k: r[k] for k in ("image", "output", "tokens", "decode_time", "status", "attempts")}
            for r in results
        ],
    }
    path = os.path.join(args.output_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def report(results: list[dict], total_jobs: int, wall_time: float) -> None:
    total_tokens = sum(r["tokens"] for r in results)
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'=' * 60}")
    print("Concurrent Results:")
    print(f"  Jobs ran: {len(results)}/{total_jobs} (ok={ok}, skipped={skipped}, failed={failed})")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time: {wall_time:.2f}s")
    if wall_time > 0:
        print(f"  System TPS: {total_tokens / wall_time:.2f} tokens/s")
    if ok > 0:
        avg_decode = sum(r["decode_time"] for r in results if r["status"] == "ok") / ok
        avg_tokens = total_tokens / ok
        print(f"  Avg tokens/request: {avg_tokens:.0f}")
        print(f"  Avg decode_time/request: {avg_decode:.2f}s")
    print(f"{'=' * 60}")


def run(args):
    jobs, tmp_dir = build_jobs(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    mode = "pdf_pages" if args.pdf else "dataset_images"
    print(f"Mode: {mode}, requests={len(jobs)}, concurrency={args.concurrency}, image_mode={args.image_mode}")

    wall_start = time.time()
    results: list[dict] = []
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {
        executor.submit(infer_one, image_path, output_file, args, i + 1): image_path
        for i, (image_path, output_file) in enumerate(jobs)
    }
    try:
        for future in as_completed(futures):
            results.append(future.result())
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — cancelling pending jobs; finished pages are saved.")
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        executor.shutdown(wait=True)
        wall_time = time.time() - wall_start
        if args.output_dir:
            manifest_path = write_manifest(args, results, wall_time)
            print(f"Manifest written: {manifest_path}")
        if tmp_dir and not args.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        report(results, len(jobs), wall_time)

    if interrupted:
        raise KeyboardInterrupt


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGLang concurrent inference for image datasets or PDF pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_dir", default="", help="Directory of images for dataset concurrency mode")
    parser.add_argument("--pdf", default="", help="PDF file; each page is converted and sent as one concurrent request")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model_dir", default="baidu/Unlimited-OCR")
    parser.add_argument("--image_mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--server_log", default="./log/sglang_server.log")
    parser.add_argument("--overwrite", action="store_true", help="Re-run pages even if their output file already exists")
    parser.add_argument("--keep_temp", action="store_true", help="Keep the temporary directory of rendered PDF pages")
    args = parser.parse_args()
    # gundam tiles a single image (crop_mode=True); multi-page/PDF inference
    # only supports base, so reject the silently-wrong combination up front.
    if args.pdf and args.image_mode != "base":
        parser.error("--pdf requires --image_mode base (gundam is for single images only)")
    return args


def main():
    args = parse_args()
    server_process = start_server(args)
    try:
        run(args)
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
