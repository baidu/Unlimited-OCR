from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .compat import patch_remote_cuda_calls
from .output import write_text_only_result
from .runtime import select_torch_device


def select_model_dtype(torch_module, device: str):
    if device == "cuda":
        return torch_module.float16
    return torch_module.float32


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = "baidu/Unlimited-OCR"
    device: str = "auto"
    prompt: str = "<image>document parsing."
    image_mode: str = "gundam"
    output_dir: Path = Path("outputs")
    max_length: int = 32768
    base_size: int = 1024
    image_size: int = 640
    no_repeat_ngram_size: int = 35
    ngram_window: int = 128
    trust_remote_code: bool = True
    write_text_result: bool = False


class UnlimitedOCRModel:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        self.device = "cpu"

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = select_torch_device(torch, self.config.device)
        dtype = select_model_dtype(torch, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            use_safetensors=True,
            torch_dtype=dtype,
        ).eval()
        self._model = self._model.to(self.device)

    def infer_image(self, image_path: Path, output_dir: Path) -> str:
        import torch

        if self._model is None or self._tokenizer is None:
            self.load()

        crop_mode = self.config.image_mode == "gundam"
        image_size = self.config.image_size if crop_mode else self.config.base_size
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch_remote_cuda_calls(torch, self.device):
            result = self._model.infer(
                self._tokenizer,
                prompt=self.config.prompt,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=self.config.base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                max_length=self.config.max_length,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                ngram_window=self.config.ngram_window,
                save_results=True,
            )
        if self.config.write_text_result:
            write_text_only_result(output_dir / "result.md")
        return "" if result is None else str(result)
