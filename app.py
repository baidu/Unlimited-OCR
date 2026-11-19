"""
Gradio demo for Hugging Face Spaces.

Provides a simple web interface for Unlimited-OCR:
  - Single image OCR via transformers pipeline
  - PDF multi-page OCR via PyMuPDF + transformers
"""

import os
import tempfile

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "baidu/Unlimited-OCR")
# Device detection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None


def load_model():
    """Load model and tokenizer once (lazy initialization)."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,  # MPS uses float32
        )
        _model = _model.eval()
        if DEVICE == "cuda":
            _model = _model.cuda()
        elif DEVICE == "mps":
            _model = _model.to("mps")
    return _model, _tokenizer


def infer_single(image: Image.Image, prompt: str) -> str:
    """Run OCR on a single PIL image."""
    model, tokenizer = load_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "input.png")
        image.save(image_path)

        result = model.infer(
            tokenizer,
            prompt=f"<image>{prompt}",
            image_file=image_path,
            output_path=tmp_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=False,
        )

    return result if isinstance(result, str) else str(result)


def infer_pdf(pdf_file, prompt: str) -> str:
    """Run OCR on an uploaded PDF file."""
    import fitz

    model, tokenizer = load_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        doc = fitz.open(pdf_file.name)
        mat = fitz.Matrix(300 / 72, 300 / 72)
        image_paths = []
        for i, page in enumerate(doc):
            out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out_path)
            image_paths.append(out_path)
        doc.close()

        result = model.infer_multi(
            tokenizer,
            prompt=f"<image>{prompt}",
            image_files=image_paths,
            output_path=tmp_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=False,
        )

    return result if isinstance(result, str) else str(result)


# ── Gradio UI ──

with gr.Blocks(
    title="Unlimited OCR Works",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 🔍 Unlimited OCR Works\n"
        "**One-shot Long-horizon Parsing** — by Baidu\n\n"
        "📄 [Paper](https://arxiv.org/abs/2606.23050) | "
        "💻 [GitHub](https://github.com/baidu/Unlimited-OCR) | "
        "🤗 [Model](https://huggingface.co/baidu/Unlimited-OCR)"
    )

    with gr.Tab("🖼️ Single Image"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image")
                img_prompt = gr.Textbox(
                    value="document parsing.",
                    label="Prompt",
                    placeholder="Enter OCR prompt...",
                )
                img_btn = gr.Button("🔍 Run OCR", variant="primary")
            with gr.Column():
                img_output = gr.Textbox(label="Result", lines=20, show_copy_button=True)

        img_btn.click(
            fn=infer_single,
            inputs=[img_input, img_prompt],
            outputs=img_output,
        )

    with gr.Tab("📄 PDF"):
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                pdf_prompt = gr.Textbox(
                    value="Multi page parsing.",
                    label="Prompt",
                    placeholder="Enter OCR prompt...",
                )
                pdf_btn = gr.Button("🔍 Run OCR", variant="primary")
            with gr.Column():
                pdf_output = gr.Textbox(label="Result", lines=20, show_copy_button=True)

        pdf_btn.click(
            fn=infer_pdf,
            inputs=[pdf_input, pdf_prompt],
            outputs=pdf_output,
        )

    gr.Markdown(
        "---\n"
        "**Notes:**\n"
        "- Single image uses `gundam` mode (base_size=1024, image_size=640, crop=True)\n"
        "- PDF uses `base` mode (image_size=1024) for multi-page parsing\n"
        "- Model loads lazily on first request — please wait ~30s for initialization"
    )


if __name__ == "__main__":
    demo.launch()
