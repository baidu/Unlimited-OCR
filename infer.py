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

PROMPT = "document parsing."
TEMPERATURE = 0
CONTEXT_LENGTH = 32768
NO_REPEAT_NGRAM_SIZE = 35
NGRAM_WINDOW = 128
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


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc):
        out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
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
        args.attention_backend,
        "--page-size",
        str(args.page_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
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
    chunks = []
    token_count = 0
    first_token_time = None
    f = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
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
    decode_time = (
        (end_time - first_token_time) if first_token_time and token_count > 1 else 0
    )
    return {"tokens": token_count, "decode_time": decode_time, "text": "".join(chunks)}


def infer_one(image_path: str, output_file: str | None, args, idx: int) -> dict:
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

    name = os.path.basename(image_path)
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
            print(
                f"  [{idx}] {name}: {result['tokens']} tokens, {result['decode_time']:.1f}s"
            )
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [{idx}] {name}: retry {attempt + 1}/{MAX_RETRIES} ({e})")
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [{idx}] {name}: FAILED ({e})")
            return {"tokens": 0, "decode_time": 0, "text": ""}


def collect_dataset_images(image_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    return sorted(image_files, key=lambda f: os.path.getsize(f), reverse=True)


def build_jobs(args) -> list[tuple[str, str | None]]:
    if args.pdf:
        image_files = pdf_to_images(args.pdf, dpi=PDF_DPI)
        prefix = os.path.splitext(os.path.basename(args.pdf))[0]
        jobs = []
        for i, image_path in enumerate(image_files):
            output_file = None
            if args.output_dir:
                output_file = os.path.join(
                    args.output_dir, f"{prefix}_page_{i + 1:04d}.md"
                )
            jobs.append((image_path, output_file))
        return jobs

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
    return jobs


def run(args):
    jobs = build_jobs(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    mode = "pdf_pages" if args.pdf else "dataset_images"
    print(
        f"Mode: {mode}, requests={len(jobs)}, concurrency={args.concurrency}, image_mode={args.image_mode}, attention_backend={args.attention_backend}"
    )

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(infer_one, image_path, output_file, args, i + 1): image_path
            for i, (image_path, output_file) in enumerate(jobs)
        }
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.time() - wall_start
    total_tokens = sum(r["tokens"] for r in results)
    successful = sum(1 for r in results if r["tokens"] > 0)
    print(f"\n{'=' * 60}")
    print("Concurrent Results:")
    print(f"  Requests: {successful}/{len(jobs)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time: {wall_time:.2f}s")
    if wall_time > 0:
        print(f"  System TPS: {total_tokens / wall_time:.2f} tokens/s")
    if successful > 0:
        avg_decode = (
            sum(r["decode_time"] for r in results if r["tokens"] > 0) / successful
        )
        avg_tokens = total_tokens / successful
        print(f"  Avg tokens/request: {avg_tokens:.0f}")
        print(f"  Avg decode_time/request: {avg_decode:.2f}s")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGLang concurrent inference for image datasets or PDF pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        default="",
        help="Directory of images for dataset concurrency mode",
    )
    parser.add_argument(
        "--pdf",
        default="",
        help="PDF file; each page is converted and sent as one concurrent request",
    )
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model_dir", default="baidu/Unlimited-OCR")
    parser.add_argument("--image_mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument(
        "--attention_backend",
        default="fa3",
        choices=("fa3", "flashinfer", "triton", "fa4", "flashmla", "cutlass"),
        help="Attention backend for SGLang. Use 'flashinfer' or 'triton' for non-Hopper GPUs (e.g. RTX 4090).",
    )
    parser.add_argument("--page_size", type=int, default=1, help="SGLang page size")
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.8,
        help="SGLang static memory fraction",
    )
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
