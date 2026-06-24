"""
Concurrent inference via SGLang.

Two input modes are supported:
  1. Dataset images: pass --image_dir and each image is sent as one request.
  2. PDF pages: pass --pdf and each converted page is sent as one request.
"""

import argparse
import base64
import json
import os
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
# Defaults follow the README's documented settings:
#   single image (gundam): ngram_size=35, ngram_window=128
#   multi-page / PDF:      ngram_size=35, ngram_window=1024
DEFAULT_NGRAM_SIZE = 35
DEFAULT_NGRAM_WINDOW = 1024
REQUEST_TIMEOUT = 1200
MAX_RETRIES = 5
NO_REPEAT_NGRAM_PROCESSOR_STR = None


def get_ngram_processor_str():
    global NO_REPEAT_NGRAM_PROCESSOR_STR
    if NO_REPEAT_NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        NO_REPEAT_NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return NO_REPEAT_NGRAM_PROCESSOR_STR


def pdf_to_images(pdf_path: str, dpi: int = 300, max_pages: int | None = None) -> tuple[list[str], tempfile.TemporaryDirectory]:
    """Render PDF pages to PNGs in a temporary directory.

    The returned TemporaryDirectory owns the tempdir; callers should keep
    it alive for the duration of inference and then let it be garbage-collected
    (or call .cleanup()) to release the rendered PNGs.
    """
    import fitz

    tmp = tempfile.TemporaryDirectory(prefix="pdf_ocr_")
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as doc:
        page_iter = doc
        if max_pages is not None:
            page_iter = list(doc)[:max_pages]
        for i, page in enumerate(page_iter):
            out_path = os.path.join(tmp.name, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out_path)
            image_paths.append(out_path)
    return image_paths, tmp


def encode_image(image_path: str) -> dict:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(image_path: str) -> list[dict]:
    return [{"type": "text", "text": PROMPT}, encode_image(image_path)]


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


def collect_stream_silent(resp, output_file: str | None, write_output: bool) -> tuple[int, float, str]:
    """Stream a server response, optionally writing the result to output_file.

    Returns (tokens, decode_time, text). When write_output is False (e.g. on
    failed requests or when --resume skipped a file), no output file is opened
    and any partial file from a previous run is left untouched.
    """
    chunks = []
    token_count = 0
    first_token_time = None
    f = open(output_file, "w", encoding="utf-8") if (output_file and write_output) else None
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

    end_time = time.time()
    decode_time = (end_time - first_token_time) if first_token_time and token_count > 1 else 0
    return token_count, decode_time, "".join(chunks)


def infer_one(
    image_path: str,
    output_file: str | None,
    args,
    idx: int,
    write_output: bool = True,
) -> dict:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": build_content(image_path)}],
        "temperature": TEMPERATURE,
        "skip_special_tokens": False,
        "stream": True,
        "images_config": {"image_mode": args.image_mode},
    }
    if args.ngram_size > 0 and args.ngram_window > 0:
        payload["custom_logit_processor"] = get_ngram_processor_str()
        payload["custom_params"] = {
            "ngram_size": args.ngram_size,
            "window_size": args.ngram_window,
        }

    name = os.path.basename(image_path)
    last_error: str | None = None
    result: dict | None = None
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
            tokens, decode_time, _text = collect_stream_silent(resp, output_file, write_output)
            print(f"  [{idx}] {name}: {tokens} tokens, {decode_time:.1f}s")
            result = {"status": "ok", "tokens": tokens, "decode_time": decode_time, "error": None}
            break
        except Exception as e:
            last_error = repr(e)
            if attempt < MAX_RETRIES - 1:
                print(f"  [{idx}] {name}: retry {attempt + 1}/{MAX_RETRIES} ({e})")
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [{idx}] {name}: FAILED ({e})")
            result = {"status": "failed", "tokens": 0, "decode_time": 0, "error": last_error}
    if result is None:
        result = {"status": "failed", "tokens": 0, "decode_time": 0, "error": "no attempts made"}
    return result


def collect_dataset_images(image_dir: str, max_images: int | None = None) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    image_files.sort()
    if max_images is not None:
        image_files = image_files[:max_images]
    return image_files


def build_jobs(args) -> tuple[list[tuple[str, str | None]], tempfile.TemporaryDirectory | None]:
    """Build (image_path, output_file) jobs.

    Returns the jobs list and an optional TemporaryDirectory that the caller
    must keep alive for the duration of inference (so the rendered PDF
    pages stay on disk). It is None for the image_dir mode.
    """
    pdf_tmp: tempfile.TemporaryDirectory | None = None
    if args.pdf:
        if args.image_mode == "gundam":
            raise ValueError(
                "--image_mode gundam is not supported with --pdf: multi-page parsing "
                "requires --image_mode base. See README."
            )
        image_files, pdf_tmp = pdf_to_images(args.pdf, dpi=PDF_DPI, max_pages=args.max_pages)
        prefix = os.path.splitext(os.path.basename(args.pdf))[0]
        jobs = []
        for i, image_path in enumerate(image_files):
            output_file = None
            if args.output_dir:
                output_file = os.path.join(args.output_dir, f"{prefix}_page_{i + 1:04d}.md")
            jobs.append((image_path, output_file))
        return jobs, pdf_tmp

    if not args.image_dir:
        raise ValueError("Either --image_dir or --pdf is required")
    image_files = collect_dataset_images(args.image_dir, max_images=args.max_images)

    jobs = []
    for image_path in image_files:
        output_file = None
        if args.output_dir:
            rel = os.path.relpath(image_path, args.image_dir)
            stem = os.path.splitext(rel)[0].replace(os.sep, "__")
            output_file = os.path.join(args.output_dir, f"{stem}.md")
        jobs.append((image_path, output_file))
    return jobs, pdf_tmp


def _already_done(output_file: str | None) -> bool:
    """Resume helper: True if the output file exists and is non-empty."""
    if not output_file:
        return False
    try:
        return os.path.getsize(output_file) > 0
    except OSError:
        return False


class ResultsWriter:
    """Appends one JSON record per request to a JSONL file."""

    def __init__(self, path: str | None):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8") if path else None

    def write(self, record: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def run(args):
    jobs, pdf_tmp = build_jobs(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    mode = "pdf_pages" if args.pdf else "dataset_images"
    print(f"Mode: {mode}, jobs={len(jobs)}, concurrency={args.concurrency}, image_mode={args.image_mode}")

    results_writer = ResultsWriter(args.results_jsonl)
    wall_start = time.time()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            skipped = 0
            for i, (image_path, output_file) in enumerate(jobs):
                if args.resume and _already_done(output_file):
                    skipped += 1
                    print(f"  [{i + 1}] {os.path.basename(image_path)}: skipped (already done)")
                    results.append({"status": "skipped", "tokens": 0, "decode_time": 0, "error": None})
                    if results_writer.path:
                        results_writer.write({
                            "index": i + 1,
                            "name": os.path.basename(image_path),
                            "status": "skipped",
                            "tokens": 0,
                            "decode_time_s": 0.0,
                            "wall_time_s": 0.0,
                            "output_file": output_file,
                            "error": None,
                        })
                    continue
                futures[executor.submit(infer_one, image_path, output_file, args, i + 1, True): (i + 1, image_path, output_file)]

            for future in as_completed(futures):
                idx, image_path, output_file = futures[future]
                request_start = time.time()
                r = future.result()
                wall = time.time() - request_start
                results.append(r)
                if results_writer.path:
                    results_writer.write({
                        "index": idx,
                        "name": os.path.basename(image_path),
                        "status": r["status"],
                        "tokens": r["tokens"],
                        "decode_time_s": r["decode_time"],
                        "wall_time_s": round(wall, 3),
                        "output_file": output_file,
                        "error": r["error"],
                    })
    finally:
        results_writer.close()
        if pdf_tmp is not None:
            pdf_tmp.cleanup()

    wall_time = time.time() - wall_start
    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] == "failed"]
    skipped_results = [r for r in results if r["status"] == "skipped"]
    total_tokens = sum(r["tokens"] for r in results)
    successful = len(ok_results)

    print(f"\n{'=' * 60}")
    print("Concurrent Results:")
    print(f"  Requests: ok={successful}, failed={len(failed_results)}, skipped={len(skipped_results)} (total {len(results)})")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time: {wall_time:.2f}s")
    if wall_time > 0:
        print(f"  System TPS: {total_tokens / wall_time:.2f} tokens/s")
    if successful > 0:
        avg_decode = sum(r["decode_time"] for r in ok_results) / successful
        avg_tokens = total_tokens / successful
        print(f"  Avg tokens/request: {avg_tokens:.0f}")
        print(f"  Avg decode_time/request: {avg_decode:.2f}s")
    if args.results_jsonl:
        print(f"  Results JSONL: {args.results_jsonl}")
    print(f"{'=' * 60}")

    # Non-zero exit on full failure so CI / pipelines can detect it.
    if jobs and successful == 0 and len(failed_results) > 0:
        sys.exit(1)


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
    parser.add_argument("--image_mode", choices=("gundam", "base"), default="base",
                        help="Use 'gundam' for single-image high-res; 'base' is required for PDF / multi-page.")
    parser.add_argument("--ngram_size", type=int, default=DEFAULT_NGRAM_SIZE,
                        help="No-repeat ngram size for the custom logit processor. 0 disables it.")
    parser.add_argument("--ngram_window", type=int, default=DEFAULT_NGRAM_WINDOW,
                        help="Ngram window size. README recommends 128 for single image, 1024 for multi-page.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images / pages whose .md already exists and is non-empty.")
    parser.add_argument("--results_jsonl", default="",
                        help="If set, write one JSON record per request to this path.")
    parser.add_argument("--max_pages", type=int, default=None,
                        help="(PDF mode) Process at most the first N pages.")
    parser.add_argument("--max_images", type=int, default=None,
                        help="(image_dir mode) Process at most the first N images.")
    parser.add_argument("--server_log", default="./log/sglang_server.log")
    return parser.parse_args()


def main():
    args = parse_args()
    server_process = start_server(args)
    try:
        run(args)
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
