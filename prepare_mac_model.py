"""
Download baidu/Unlimited-OCR from Hugging Face and patch it for macOS (MPS or CPU).

The upstream model code hard-codes CUDA: it calls .cuda() on tensors and wraps generation
in torch.autocast("cuda", dtype=torch.bfloat16). On a Mac that fails before any compute
runs. This script:

  1. Downloads the model snapshot from Hugging Face (uses the HF cache; re-runs are cheap).
  2. Mirrors it into ./model_local/, with .py / .json files copied locally and the big
     .safetensors weights symlinked back to the HF cache (so we don't duplicate ~6.7 GB).
  3. Edits modeling_unlimitedocr.py:
       - replaces every .cuda() on input/image tensors with .to(self.device)
       - replaces torch.autocast("cuda", ...) with a no-op context on non-CUDA devices
       - replaces the hard-coded .to(torch.bfloat16) on image tensors with .to(self.dtype)

After running this, point infer_mac.py at ./model_local/ and pick --device mps or cpu.
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "baidu/Unlimited-OCR"
HERE = Path(__file__).resolve().parent
LOCAL_DIR = HERE / "model_local"
TARGET_PY = "modeling_unlimitedocr.py"

PATCH_HEADER = '''\
import contextlib as _contextlib

def _mac_autocast(device):
    # No-op on non-CUDA devices; model is already loaded in the target dtype on MPS/CPU.
    if getattr(device, "type", None) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return _contextlib.nullcontext()


'''

REPLACEMENTS = [
    ("input_ids.unsqueeze(0).cuda()",
     "input_ids.unsqueeze(0).to(self.device)"),
    ("images_seq_mask.unsqueeze(0).cuda()",
     "images_seq_mask.unsqueeze(0).to(self.device)"),
    ("images=[(images_crop.cuda(), images_ori.cuda())],",
     "images=[(images_crop.to(self.device), images_ori.to(self.device))],"),
    ("images=[(dummy_crop.cuda(), images_ori.cuda())],",
     "images=[(dummy_crop.to(self.device), images_ori.to(self.device))],"),
    ("images_seq_mask[idx].unsqueeze(-1).cuda()",
     "images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device)"),
    ('with torch.autocast("cuda", dtype=torch.bfloat16):',
     "with _mac_autocast(self.device):"),
    ("image_transform(global_view).to(torch.bfloat16)",
     "image_transform(global_view).to(self.dtype)"),
    ("image_transform(images_crop_raw[i]).to(torch.bfloat16)",
     "image_transform(images_crop_raw[i]).to(self.dtype)"),
]


def mirror_snapshot(snap_dir: Path) -> None:
    LOCAL_DIR.mkdir(exist_ok=True)
    for src in snap_dir.iterdir():
        dst = LOCAL_DIR / src.name
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir() or src.suffix in (".safetensors", ".pdf", ".png", ".gif"):
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)


def patch_modeling() -> None:
    path = LOCAL_DIR / TARGET_PY
    src = path.read_text(encoding="utf-8")

    if "_mac_autocast" in src:
        print(f"  {TARGET_PY} already patched, skipping")
        return

    anchor = "import numpy as np\nimport time\n"
    if anchor not in src:
        raise RuntimeError(
            f"expected anchor not found in {TARGET_PY}; upstream model code may have changed"
        )
    src = src.replace(anchor, anchor + "\n" + PATCH_HEADER, 1)

    for old, new in REPLACEMENTS:
        count = src.count(old)
        if count == 0:
            print(f"  warning: pattern not found (upstream may have changed): {old[:60]}...")
            continue
        src = src.replace(old, new)
        print(f"  replaced {count}x: {old[:70]}...")

    path.write_text(src, encoding="utf-8")


def main():
    print(f"[1/3] Downloading {REPO_ID} (uses HF cache; ~6.7 GB on first run)...")
    snap = Path(snapshot_download(repo_id=REPO_ID))
    print(f"      snapshot at {snap}")

    print(f"[2/3] Mirroring into {LOCAL_DIR}...")
    mirror_snapshot(snap)

    print(f"[3/3] Patching {TARGET_PY} for MPS/CPU...")
    patch_modeling()

    print(f"\nReady. Run:  python infer_mac.py <pdf>  [--device mps|cpu] [--dtype fp32|bf16]")


if __name__ == "__main__":
    main()
