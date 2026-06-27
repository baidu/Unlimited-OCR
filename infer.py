"""
Concurrent inference via SGLang.

Two input modes are supported:
  1. Dataset images: pass --image_dir and each image is sent as one request.
  2. PDF pages: pass --pdf and each converted page is sent as one request.
"""

import typer
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import requests
from tqdm import tqdm

SERVED_MODEL_NAME = "Unlimited-OCR"
SERVER_URL = "http://127.0.0.1:27100"
HOST = "0.0.0.0"
PORT = 27100
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


def get_ngram_processor_str():
    global NO_REPEAT_NGRAM_PROCESSOR_STR
    if NO_REPEAT_NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        NO_REPEAT_NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return NO_REPEAT_NGRAM_PROCESSOR_STR


def parse_line(line: str):
    # Matches: <|det|>tag_name [xmin, ymin, xmax, ymax]<|/det|>content
    match = re.match(r'^<\|det\|>([a-zA-Z_0-9]+)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]<\|/det\|>(.*)', line, re.DOTALL)
    if match:
        tag_name = match.group(1)
        bbox = [int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))]
        content = match.group(6)
        return tag_name, bbox, content
    return None, None, line


def determine_title_level(text: str) -> int:
    text_clean = text.strip()
    # Level 1: Chapters, Appendices, References, Exercises, etc.
    if re.match(r'^(第[0-9一二三四五六七八九十百]+[章篇]|习题|参考文献|目录|前言|序言|后记|索引|主要符号表|内容简介|附录)', text_clean):
        return 1
    if re.match(r'^[A-Z]\s+[\u4e00-\u9fa5]+', text_clean):
        return 1
        
    # Level 2: "1.1 引言", "A.1 基本演算"
    if re.match(r'^([A-Z]|\d+)\.\d+(\s+|$)', text_clean):
        return 2
        
    # Level 3: "10.5.1 等度量映射", "C.1.1 均匀分布"
    if re.match(r'^([A-Z]|\d+)\.\d+\.\d+(\s+|$)', text_clean):
        return 3
        
    # Level 4: "10.5.1.1"
    if re.match(r'^([A-Z]|\d+)\.\d+\.\d+\.\d+(\s+|$)', text_clean):
        return 4
        
    # Lists treated as Level 4
    if text_clean.startswith('-') or text_clean.startswith('*'):
        return 4
        
    return 3


def format_title(text: str) -> str:
    level = determine_title_level(text)
    text_clean = text.strip()
    if level == 4 and (text_clean.startswith('-') or text_clean.startswith('*')):
        text_clean = text_clean[1:].strip()
    return "#" * level + " " + text_clean


def convert_raw_to_markdown(raw_text: str, image_path: str, output_file: str) -> str:
    lines = raw_text.split('\n')
    blocks = []
    image_counter = 0
    
    output_dir = os.path.dirname(os.path.abspath(output_file)) if output_file else "."
    attachments_dir = os.path.join(output_dir, "attachments")
    page_stem = os.path.splitext(os.path.basename(output_file))[0] if output_file else "page"
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        tag_name, bbox, content_val = parse_line(line)
        if not tag_name:
            if line_stripped:
                blocks.append(line_stripped)
            continue
            
        content_stripped = content_val.strip()
        
        if tag_name == 'page_number':
            blocks.append(f"<!-- page {content_stripped} -->")
        elif tag_name in ('header', 'footer', 'aside_text'):
            blocks.append(f"<!-- {tag_name}: {content_stripped} -->")
        elif tag_name == 'title':
            blocks.append(format_title(content_stripped))
        elif tag_name in ('image', 'chart'):
            crop_filename = f"{page_stem}_{tag_name}_{image_counter}.png"
            if os.path.exists(image_path):
                os.makedirs(attachments_dir, exist_ok=True)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        xmin, ymin, xmax, ymax = bbox
                        
                        left = int(round(xmin * width / 1000.0))
                        top = int(round(ymin * height / 1000.0))
                        right = int(round(xmax * width / 1000.0))
                        bottom = int(round(ymax * height / 1000.0))
                        
                        left = max(0, min(left, width - 1))
                        top = max(0, min(top, height - 1))
                        right = max(left + 1, min(right, width))
                        bottom = max(top + 1, min(bottom, height))
                        
                        cropped = img.crop((left, top, right, bottom))
                        cropped.save(os.path.join(attachments_dir, crop_filename))
                        blocks.append(f"![](attachments/{crop_filename})")
                        image_counter += 1
                except Exception as e:
                    print(f"Error cropping {tag_name} in {output_file}: {e}")
                    blocks.append(f"![](attachments/{crop_filename})")
                    image_counter += 1
            else:
                blocks.append(f"![](attachments/{crop_filename})")
                image_counter += 1
        elif tag_name in ('text', 'ref_text', 'image_caption', 'table', 'equation'):
            if content_stripped:
                blocks.append(content_stripped)
        else:
            if content_stripped:
                blocks.append(content_stripped)
                
    return "\n\n".join(blocks)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    total = len(doc)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(tqdm(doc, total=total, desc="Converting PDF", unit="page", leave=False)):
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
    global PORT, SERVER_URL
    port = args.port
    server_url = f"http://127.0.0.1:{port}"

    if server_ready(server_url):
        PORT = port
        SERVER_URL = server_url
        print(f"Reuse existing SGLang server: {SERVER_URL}")
        return None

    # Check if the desired port is occupied
    import socket
    port_is_free = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
        port_is_free = True
    except OSError:
        pass

    if not port_is_free:
        # Find the next available free port
        new_port = port + 1
        while new_port < 65535:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', new_port))
                port = new_port
                server_url = f"http://127.0.0.1:{port}"
                print(f"Port {args.port} is occupied. Automatically switching to free port: {port}")
                break
            except OSError:
                new_port += 1
        else:
            raise RuntimeError("No free ports available")

    PORT = port
    SERVER_URL = server_url

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
    num_gpus = len(args.gpu.split(","))
    if num_gpus > 1:
        cmd.extend(["--tp-size", str(num_gpus)])

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
    return {"tokens": token_count, "decode_time": decode_time, "text": "".join(chunks)}


def infer_one(image_path: str, output_file: str | None, args, idx: int, server_url: str | None = None) -> dict:
    url = server_url or SERVER_URL
    payload = {
        "model": args.model if hasattr(args, 'model') and args.model else SERVED_MODEL_NAME,
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
                f"{url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            if resp.status_code == 502 and attempt < MAX_RETRIES - 1:
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            result = collect_stream_silent(resp, None)
            if output_file:
                formatted_text = convert_raw_to_markdown(result["text"], image_path, output_file)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
            print(f"  [{idx}] {name}: {result['tokens']} tokens, {result['decode_time']:.1f}s")
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
                output_file = os.path.join(args.output_dir, f"{prefix}_page_{i + 1:04d}.md")
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


def run(args, server_url: str | None = None):
    if args.output_dir == "./outputs":
        if args.pdf:
            pdf_stem = os.path.splitext(os.path.basename(args.pdf))[0]
            args.output_dir = os.path.join("./outputs", pdf_stem)
        elif args.image_dir:
            image_dir_stem = os.path.basename(os.path.normpath(args.image_dir))
            args.output_dir = os.path.join("./outputs", image_dir_stem)

    jobs = build_jobs(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    mode = "pdf_pages" if args.pdf else "dataset_images"
    total = len(jobs)
    print(f"Mode: {mode}, total={total}, concurrency={args.concurrency}, image_mode={args.image_mode}")

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(infer_one, image_path, output_file, args, i + 1, server_url): image_path
            for i, (image_path, output_file) in enumerate(jobs)
        }
        pbar = tqdm(as_completed(futures), total=total, desc="Inferring", unit="page", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for future in pbar:
            result = future.result()
            results.append(result)
            pbar.set_postfix_str(
                f"tokens={result['tokens']} decode={result['decode_time']:.1f}s"
            )
        pbar.close()

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
        avg_decode = sum(r["decode_time"] for r in results if r["tokens"] > 0) / successful
        avg_tokens = total_tokens / successful
        print(f"  Avg tokens/request: {avg_tokens:.0f}")
        print(f"  Avg decode_time/request: {avg_decode:.2f}s")
    print(f"{'=' * 60}")


app = typer.Typer()


class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@app.command()
def main(
    image_dir: str = typer.Option("", help="Directory of images for dataset concurrency mode"),
    pdf: str = typer.Option("", help="PDF file; each page is converted and sent as one concurrent request"),
    output_dir: str = typer.Option("./outputs", help="Output directory for markdown and cropped images"),
    concurrency: int = typer.Option(8, help="Concurrency limit"),
    gpu: str = typer.Option("0", help="GPU devices"),
    model_dir: str = typer.Option("baidu/Unlimited-OCR", help="Model directory path"),
    image_mode: str = typer.Option("gundam", help="Image mode: gundam or base"),
    server_log: str = typer.Option("./log/sglang_server.log", help="Log file for SGLang server"),
    port: int = typer.Option(27100, help="SGLang server port"),
    api_url: str = typer.Option("", help="Custom API URL (e.g. https://api.openai.com/v1). When set, no local server is started."),
    model: str = typer.Option("", help="Model name for custom API (default: Unlimited-OCR)"),
):
    args = ArgsNamespace(
        image_dir=image_dir,
        pdf=pdf,
        output_dir=output_dir,
        concurrency=concurrency,
        gpu=gpu,
        model_dir=model_dir,
        image_mode=image_mode,
        server_log=server_log,
        port=port,
        api_url=api_url,
        model=model,
    )
    if args.api_url:
        server_url = args.api_url.rstrip("/")
        print(f"Using custom API: {server_url}")
        run(args, server_url=server_url)
    else:
        server_process = start_server(args)
        try:
            run(args)
        finally:
            stop_server(server_process)


if __name__ == "__main__":
    app()
