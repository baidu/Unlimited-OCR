#!/usr/bin/env python3
"""Step 1 of the evaluation: send documents to the SGLang server and save the raw outputs.

Each document is a directory of ``page_<n>.(jpg|jpeg|png)`` images. The whole document goes
out as one request: the images in page order, then ``"<PAGE><image>" * n_pages +
"Multi page parsing."``. The raw output is written to ``<raw-dir>/<doc>.txt``. Run
``utils/process_unlimitedOCR2.py`` on it to get markdown.

Token budget (same as our evaluation setup):
  * gundam mode (``crop_mode=True``) by default;
  * if the benchmark ships a per-pdf token table (``<root>/<name>_per_pdf_{gundam,base}.jsonl``),
    ``max_tokens = max_model_len - image_tokens - 700``. A multi-page document whose budget
    is below its ground-truth length falls back to base mode;
  * without the table, ``max_tokens = 0.3 * max_model_len``. If the server rejects the
    request as too long, ``max_tokens`` is refit once from the prompt length the server
    reports.

Examples:
  python infer.py --doc demo --out raw_demo.txt
  python infer.py --benchmark MPDocBench --dataset-root /path/to/MPDocBench --all \\
      --raw-dir raw --concurrency 4
"""
import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

from repetition_guard import is_looping_output, truncate_looping_output

# max_model_len per benchmark; gen_reserve = 0.3 * max_model_len.
BENCHMARKS = {'MPDocBench': 131072, 'MarathonDocBench': 786432}
MODEL = 'Unlimited-OCR-2'
PROMPT_MARGIN = 700
NGRAM_SIZE = 35
NGRAM_PROCESSOR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
RE_SGLANG_COUNTS = re.compile(
    r'total of\s+(\d+)\s+tokens:\s*(\d+)\s+tokens from the input messages'
    r'\s*and\s*(\d+)\s+tokens for the completion', re.I)


def list_pages(doc_dir, max_pages):
    if not os.path.isdir(doc_dir):
        raise SystemExit(f'doc dir not found: {doc_dir}')
    files = [f for f in os.listdir(doc_dir) if re.match(r'page_\d+\.(jpg|jpeg|png)$', f, re.I)]
    files.sort(key=lambda f: int(re.search(r'page_(\d+)', f).group(1)))
    if max_pages:
        files = files[:max_pages]
    return os.path.basename(os.path.abspath(doc_dir)), [os.path.join(doc_dir, f) for f in files]


def per_pdf_tokens(dataset_root, key, gundam):
    """(image_tokens, gt_text_tokens) from the per-pdf table, or (None, None) if unavailable."""
    if not dataset_root:
        return None, None
    name = os.path.basename(dataset_root.rstrip('/'))
    path = os.path.join(dataset_root, f'{name}_per_pdf_{"gundam" if gundam else "base"}.jsonl')
    if not os.path.exists(path):
        return None, None
    with open(path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if os.path.basename(d.get('pdf', '')) == key:
                return int(d['image_tokens']), int(d['text_tokens'])
    return None, None


def plan(dataset_root, key, n_pages, max_model_len):
    """Return (crop_mode, max_tokens)."""
    gen_reserve = int(max_model_len * 0.3)
    crop_mode = True
    for _ in range(2):
        img_tok, gt_tok = per_pdf_tokens(dataset_root, key, crop_mode)
        if img_tok is None:  # not in the table: rely on the server-side refit
            return crop_mode, gen_reserve
        avail = max_model_len - img_tok - PROMPT_MARGIN
        need = gt_tok + 1 if gt_tok is not None else gen_reserve
        if avail < need and crop_mode and n_pages > 1:
            print(f'[MODE] {key}: avail={avail} < need={need}, fall back to base')
            crop_mode = False
            continue
        return crop_mode, max(1, avail)
    return crop_mode, gen_reserve


def build_payload(pages, crop_mode, max_tokens):
    content = []
    for p in pages:
        with open(p, 'rb') as fp:
            b64 = base64.b64encode(fp.read()).decode('utf-8')
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    content.append({'type': 'text', 'text': '<PAGE><image>' * len(pages) + 'Multi page parsing.'})
    window = 1024 if len(pages) > 1 else 128
    return {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': max_tokens,
        'temperature': 0.0,
        'skip_special_tokens': False,
        'images_config': {'crop_mode': crop_mode},  # True = gundam (tiles), False = base
        'custom_logit_processor': NGRAM_PROCESSOR,
        'custom_params': {'ngram_size': NGRAM_SIZE, 'window_size': window},
    }


def request_one(url, doc_dir, args):
    """Request one document; returns (stats, raw_answer). Raises SystemExit on failure."""
    key, pages = list_pages(doc_dir, args.max_pages)
    crop_mode, max_tokens = plan(args.dataset_root, key, len(pages), args.max_model_len)
    payload = build_payload(pages, crop_mode, max_tokens)
    for attempt in range(2):
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=args.timeout,
                             headers={'Content-Type': 'application/json'})
        if resp.status_code != 200:
            body = resp.text[:1000]
            m = RE_SGLANG_COUNTS.search(body)
            if m and attempt == 0:  # too long: refit from the real prompt length once
                payload['max_tokens'] = max(1, args.max_model_len - int(m.group(2)) - 64)
                print(f"[REFIT] {key}: input={m.group(2)} -> max_tokens={payload['max_tokens']}")
                continue
            raise SystemExit(f'HTTP {resp.status_code} {body}')
        data = resp.json()
        choice = data['choices'][0]
        answer = choice['message'].get('content') or ''
        loop_detected = loop_truncated = False
        if args.loop_guard:
            if is_looping_output(answer):
                loop_detected = True
                guarded = truncate_looping_output(answer)
                loop_truncated = guarded != answer
                answer = guarded
        usage = data.get('usage') or {}
        stats = {
            'doc': key, 'pages': len(pages), 'crop_mode': crop_mode,
            'max_tokens': payload['max_tokens'], 'finish_reason': choice.get('finish_reason'),
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'page_markers': answer.count('<PAGE>'), 'seconds': round(time.time() - t0, 1),
        }
        if args.loop_guard:  # machine-readable incomplete-output status
            stats['loop_detected'] = loop_detected
            stats['loop_truncated'] = loop_truncated
        return stats, answer
    raise SystemExit(f'{key}: still failing after refit')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='127.0.0.1:10000', help='SGLang server host:port')
    ap.add_argument('--benchmark', choices=list(BENCHMARKS), default='MPDocBench',
                    help='sets max_model_len (MPDocBench=131072, MarathonDocBench=786432)')
    ap.add_argument('--max-model-len', type=int, default=None, help='override max_model_len')
    ap.add_argument('--dataset-root', default=None,
                    help='benchmark root (<root>/images/<doc>/page_*.jpg + per-pdf token tables)')
    ap.add_argument('--doc', default=None, help='a document directory, or a doc id under <root>/images')
    ap.add_argument('--all', action='store_true', help='request every document under <root>/images')
    ap.add_argument('--max-pages', type=int, default=0, help='only send the first N pages, 0 = all')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=6000)
    ap.add_argument('--raw-dir', default=None, help='write raw outputs to <raw-dir>/<doc>.txt')
    ap.add_argument('--loop-guard', action='store_true',
                    help='check each raw output for pathological repetition (issue #55); '
                         'truncates it at the repetition onset with an explicit marker and '
                         'reports loop_detected/loop_truncated in the stats line')
    ap.add_argument('--out', default=None, help='single document: write the raw output here')
    args = ap.parse_args()

    args.max_model_len = args.max_model_len or BENCHMARKS[args.benchmark]
    images_root = os.path.join(args.dataset_root, 'images') if args.dataset_root else None
    if args.all:
        if not images_root or not args.raw_dir:
            raise SystemExit('--all needs --dataset-root and --raw-dir')
        docs = [os.path.join(images_root, d) for d in sorted(os.listdir(images_root))]
    elif args.doc:
        docs = [args.doc if os.path.isdir(args.doc) or not images_root
                else os.path.join(images_root, args.doc)]
    else:
        raise SystemExit('give --doc or --all')
    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
    url = f"http://{args.url.replace('http://', '').rstrip('/')}/v1/chat/completions"
    print(f'[CFG] url={url} max_model_len={args.max_model_len} docs={len(docs)}', flush=True)

    def handle(doc_dir):
        stats, answer = request_one(url, doc_dir, args)
        if args.raw_dir:
            with open(os.path.join(args.raw_dir, stats['doc'] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(answer)
        return stats, answer

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(args.concurrency, len(docs)))) as ex:
        futures = {ex.submit(handle, d): d for d in docs}
        for fut in as_completed(futures):
            done += 1
            try:
                stats, answer = fut.result()
            except (Exception, SystemExit) as e:  # one failed document must not stop the run
                print(json.dumps({'doc': futures[fut], 'error': str(e)[:300],
                                  'progress': f'{done}/{len(docs)}'}, ensure_ascii=False))
                continue
            stats['progress'] = f'{done}/{len(docs)}'
            print(json.dumps(stats, ensure_ascii=False), flush=True)
            if len(docs) == 1 and not args.raw_dir:
                if args.out:
                    with open(args.out, 'w', encoding='utf-8') as f:
                        f.write(answer)
                    print('saved ->', args.out)
                else:
                    print(answer)


if __name__ == '__main__':
    main()
