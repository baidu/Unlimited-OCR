"""Patch a local baidu/Unlimited-OCR snapshot for device-agnostic inference.

The released ``modeling_unlimitedocr.py`` hardcodes CUDA (``.cuda()`` calls and
``torch.autocast("cuda", ...)``), so the model cannot run on Apple Silicon (MPS)
or CPU at all. Replacing ``.cuda()`` with ``.to(device)`` is NOT sufficient: on
the MPS backend ``torch.Tensor.masked_scatter_`` silently mis-scatters when its
mask is a stride-0 broadcast view (which the ``.unsqueeze(-1)`` at the injection
call produces) or when its source tensor is non-contiguous. Either trigger
scrambles the injected image embeddings and makes the model emit an immediate
end-of-sequence token (empty output, no error); the broadcast mask is the one
live at this call site, since the source comes from ``torch.cat`` and is
contiguous. This script therefore applies two kinds of edits:

  1. Replace the image-embedding injection with explicit positional assignment,
     which is mathematically identical and correct on CUDA, MPS, and CPU.
  2. Make every hardcoded ``.cuda()`` / ``autocast("cuda")`` follow the model's
     actual device (a no-op change on CUDA machines).

Usage:
    hf download baidu/Unlimited-OCR --local-dir ./Unlimited-OCR-local
    python patch_model_for_local.py ./Unlimited-OCR-local

The script asserts on the exact released code before editing, so it fails
loudly instead of mis-patching if the upstream file changes, and it is a no-op
when the snapshot is already patched.
"""

import pathlib
import sys

INJECTION_BEFORE = (
    "                    inputs_embeds[idx].masked_scatter_("
    "images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)"
)
INJECTION_AFTER = """\
                    # masked_scatter_ silently mis-scatters on the MPS backend
                    # when its mask is a stride-0 broadcast view (as the
                    # .unsqueeze(-1) here produced) or its source tensor is
                    # non-contiguous; explicit positional assignment computes
                    # the same result and is correct on CUDA, MPS, and CPU.
                    _feat = images_in_this_batch.to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                    _pos = (
                        images_seq_mask[idx]
                        .to(inputs_embeds.device)
                        .bool()
                        .nonzero(as_tuple=True)[0]
                    )
                    inputs_embeds[idx, _pos] = _feat"""

DEVICE_REPLACEMENTS = [
    # (pattern, replacement, expected occurrences)
    (".unsqueeze(0).cuda().shape[1]", ".unsqueeze(0).shape[1]", 4),
    ("input_ids.unsqueeze(0).cuda()", "input_ids.unsqueeze(0).to(self.device)", 3),
    ("images_seq_mask.unsqueeze(0).cuda()",
     "images_seq_mask.unsqueeze(0).to(self.device)", 3),
    ("images_crop.cuda()", "images_crop.to(self.device)", 2),
    ("images_ori.cuda()", "images_ori.to(self.device)", 3),
    ("dummy_crop.cuda()", "dummy_crop.to(self.device)", 1),
    ('with torch.autocast("cuda", dtype=torch.bfloat16):',
     'with torch.autocast(self.device.type, dtype=torch.bfloat16, '
     'enabled=(self.device.type == "cuda")):', 3),
]


def patch_file(path: pathlib.Path) -> None:
    src = path.read_text(encoding="utf-8")

    if "inputs_embeds[idx, _pos] = _feat" in src and ".cuda()" not in src:
        print(f"{path.name}: already patched, nothing to do.")
        return

    assert "class UnlimitedOCRForCausalLM" in src, "unexpected file contents"

    assert INJECTION_BEFORE in src, (
        "expected released injection line not found; the upstream file may "
        "have changed -- refusing to patch blindly"
    )
    src = src.replace(INJECTION_BEFORE, INJECTION_AFTER)

    for pattern, replacement, expected in DEVICE_REPLACEMENTS:
        found = src.count(pattern)
        assert found == expected, (
            f"expected {expected} occurrence(s) of {pattern!r}, found {found}"
        )
        src = src.replace(pattern, replacement)

    assert ".cuda()" not in src, "unreplaced .cuda() call remains"
    path.write_text(src, encoding="utf-8")
    print(f"{path.name}: patched (1 injection fix, "
          f"{sum(n for _, _, n in DEVICE_REPLACEMENTS)} device edits).")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python patch_model_for_local.py <model_snapshot_dir>")
    model_dir = pathlib.Path(sys.argv[1])
    target = model_dir / "modeling_unlimitedocr.py"
    if not target.exists():
        sys.exit(f"{target} not found -- is this an Unlimited-OCR snapshot?")
    patch_file(target)


if __name__ == "__main__":
    main()
