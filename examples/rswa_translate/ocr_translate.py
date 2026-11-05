"""
ocr_translate.py — 端到端示例：百度 Unlimited-OCR（R-SWA 文档解析）
                  + R-SWA 增强的 Qwen2.5 翻译模型。

演示 R-SWA 的通用性：百度 Unlimited-OCR 用 R-SWA 做长文档一次性解析，
本示例把它的输出（带版式 markdown）再喂给一个用 R-SWA 增强的 Qwen2.5
翻译模型，完成“文档解析 → 翻译”的长链路，且翻译侧 KV Cache 恒定有界。

    图片/PDF ──[Unlimited-OCR (R-SWA)]──> 带版式文本
                                      │
                                      ▼ 分段
                           [R-SWA Qwen2.5 翻译] 逐段翻译
                                      │
                                      ▼ 拼回
                               带版式译文

亮点：
    - 解析侧：Unlimited-OCR 借 R-SWA 一次读完数十页，KV 不随页增长。
    - 翻译侧：R-SWA 把译文 KV 限制在 prefill_len + W，长文档不爆显存。
    - 源文全程可见：跨段术语/指代一致性保真（普通滑窗会丢上下文）。

依赖：transformers>=4.57.1, torch, 以及本目录下的 rswa_qwen25.py。
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from text_utils import build_translate_prompt, segment_text
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1) OCR：复用百度 Unlimited-OCR（底层即 R-SWA）
# ---------------------------------------------------------------------------
OCR_MODEL = os.environ.get("UOCR_MODEL", "baidu/Unlimited-OCR")


def load_ocr():
    tok = AutoTokenizer.from_pretrained(OCR_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        OCR_MODEL,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    return tok, model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def ocr_image(model, tokenizer, image_file, output_dir=None, max_length=32768):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="uocr_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    outputs = model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=image_file,
        output_path=output_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        max_length=max_length,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=output_dir is not None,
    )
    return outputs if isinstance(outputs, str) else str(outputs)


# ---------------------------------------------------------------------------
# 2) R-SWA 翻译模型（Qwen2.5 + 环形滑窗注意力）
#    机制移植自本仓库的 rswa_qwen25.py（源自百度 Unlimited-OCR 的
#    SlidingWindowLlamaAttention 思想，搬到 Qwen2 架构）。
# ---------------------------------------------------------------------------
def load_translator(window=512, model_name=None):
    from rswa_qwen25 import patch_rswa

    name = model_name or os.environ.get("RSWA_QWEN25", "Qwen/Qwen2.5-0.5B")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # R-SWA 实验臂；window=None 退化为全注意力（对照臂）
    patch_rswa(model, window=window)
    return model.eval(), tok


# ---------------------------------------------------------------------------
# 3) 分段 + 翻译
#    segment_text 已抽到同目录的 text_utils.py（无 torch 依赖，便于单测）。
# ---------------------------------------------------------------------------


@torch.no_grad()
def translate_segment(model, tokenizer, segment, tgt_lang="Chinese", max_new_tokens=256):
    dev = next(model.parameters()).device
    prompt = build_translate_prompt(segment, tgt_lang=tgt_lang)
    ids = tokenizer(prompt, return_tensors="pt").to(dev)
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    trans = gen[len(tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)) :].strip()
    return trans


# ---------------------------------------------------------------------------
# 4) 端到端流水线
# ---------------------------------------------------------------------------
def run_pipeline(
    image_file,
    window=512,
    tgt_lang="Chinese",
    output_dir=None,
    ocr_max_length=32768,
    max_new_tokens=256,
):
    tok_o, model_o = load_ocr()
    source_text = ocr_image(
        model_o, tok_o, image_file, output_dir=output_dir, max_length=ocr_max_length
    )
    del model_o, tok_o
    torch.cuda.empty_cache()

    model_t, tok_t = load_translator(window=window)
    segs = segment_text(source_text)
    translations = [
        translate_segment(model_t, tok_t, s, tgt_lang=tgt_lang, max_new_tokens=max_new_tokens)
        for s in segs
    ]
    translated = "\n\n".join(translations)

    if output_dir:
        with open(os.path.join(output_dir, "source.md"), "w", encoding="utf-8") as f:
            f.write(source_text)
        with open(os.path.join(output_dir, "translated.md"), "w", encoding="utf-8") as f:
            f.write(translated)

    return {
        "source_text": source_text,
        "segments": segs,
        "translations": translations,
        "translated": translated,
        "meta": {"n_segments": len(segs), "tgt_lang": tgt_lang, "window": window},
    }


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "examples/images/sample.png"
    out = run_pipeline(img, window=512, output_dir="outputs")
    print(f"[pipeline] {out['meta']['n_segments']} segments translated.")
    print(out["translated"][:500])
