"""Unit tests for the local-inference patch (patch_model_for_local.py).

Covers the things the patch must guarantee:
  1. The positional-assignment replacement computes exactly what the released
     masked_scatter_ injection computes (checked against CPU, where
     masked_scatter_ is correct), for contiguous AND non-contiguous sources.
  2. The same equivalence for the injection call's exact real layout: a
     [B, T, H] destination written through a batch-index view, the [T, 1]
     stride-0 broadcast mask produced by .unsqueeze(-1), and a
     torch.cat-built (contiguous) source.
  3. On the MPS backend, the replacement matches the CPU reference. (The
     motivation is that raw masked_scatter_ mis-scatters there under two
     independent conditions: a stride-0 broadcast mask -- the trigger that
     actually fires at the injection call -- or a non-contiguous source.
     The tests assert our replacement is right rather than asserting the
     upstream op is wrong, so they stay green if PyTorch fixes the
     underlying issue.)

Run:  pytest tests/test_local_patch.py
"""

import torch


def _reference_injection(dst, mask, src):
    """The released behavior: in-place masked_scatter_ (computed on CPU)."""
    out = dst.clone().cpu()
    out.masked_scatter_(mask.cpu().unsqueeze(-1), src.cpu().contiguous())
    return out


def _patched_injection(dst, mask, src):
    """The patched behavior: explicit positional assignment (any device)."""
    out = dst.clone()
    feat = src.to(device=out.device, dtype=out.dtype)
    pos = mask.to(out.device).bool().nonzero(as_tuple=True)[0]
    out[pos] = feat
    return out


def _make_case(seq_len=32, hidden=16, n_img=10, noncontiguous=False, device="cpu"):
    torch.manual_seed(0)
    dst = torch.randn(seq_len, hidden, device=device)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[3:3 + n_img] = True
    if noncontiguous:
        src = torch.randn(hidden, n_img, device=device).T  # transposed view
        assert not src.is_contiguous()
    else:
        src = torch.randn(n_img, hidden, device=device)
    return dst, mask, src


def test_positional_assignment_matches_masked_scatter_contiguous():
    dst, mask, src = _make_case(noncontiguous=False)
    assert torch.equal(_patched_injection(dst, mask, src),
                       _reference_injection(dst, mask, src))


def test_positional_assignment_matches_masked_scatter_noncontiguous():
    dst, mask, src = _make_case(noncontiguous=True)
    assert torch.equal(_patched_injection(dst, mask, src),
                       _reference_injection(dst, mask, src))


def test_patched_injection_correct_on_mps():
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    for noncontiguous in (False, True):
        dst, mask, src = _make_case(noncontiguous=noncontiguous, device="mps")
        got = _patched_injection(dst, mask, src).cpu()
        want = _reference_injection(dst, mask, src)
        assert torch.allclose(got, want), (
            f"patched injection wrong on MPS (noncontiguous={noncontiguous})"
        )


def _make_real_call_case(batch=2, seq_len=32, hidden=16, device="cpu",
                         dtype=torch.float32):
    """The injection call's exact layout: a [B, T, H] batch tensor written
    through a batch-index view, a [B, T] mask whose row the released code
    broadcasts via .unsqueeze(-1) (stride 0 across the hidden dim), and
    per-row sources built with torch.cat like images_in_this_batch
    (contiguous by construction)."""
    torch.manual_seed(0)
    emb = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    mask[0, 3:3 + 6] = True
    mask[1, 10:10 + 4] = True
    srcs = [
        torch.cat([torch.randn(4, hidden, device=device, dtype=dtype),
                   torch.randn(2, hidden, device=device, dtype=dtype)], dim=0),
        torch.cat([torch.randn(3, hidden, device=device, dtype=dtype),
                   torch.randn(1, hidden, device=device, dtype=dtype)], dim=0),
    ]
    for src in srcs:
        assert src.is_contiguous()
    return emb, mask, srcs


def _reference_real_call(emb, mask, srcs):
    """The released behavior, computed on CPU: per-row in-place
    masked_scatter_ with the stride-0 broadcast mask from .unsqueeze(-1)."""
    out = emb.clone().cpu()
    for idx, src in enumerate(srcs):
        out[idx].masked_scatter_(mask[idx].cpu().unsqueeze(-1), src.cpu())
    return out


def _patched_real_call(emb, mask, srcs):
    """The patched behavior, exactly as patch_model_for_local.py writes it."""
    out = emb.clone()
    for idx, src in enumerate(srcs):
        _feat = src.to(device=out.device, dtype=out.dtype)
        _pos = mask[idx].to(out.device).bool().nonzero(as_tuple=True)[0]
        out[idx, _pos] = _feat
    return out


def test_positional_assignment_matches_broadcast_mask_layout():
    emb, mask, srcs = _make_real_call_case()
    assert torch.equal(_patched_real_call(emb, mask, srcs),
                       _reference_real_call(emb, mask, srcs))


def test_broadcast_mask_layout_correct_on_mps():
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    for dtype in (torch.float32, torch.bfloat16):
        emb, mask, srcs = _make_real_call_case(device="mps", dtype=dtype)
        got = _patched_real_call(emb, mask, srcs).cpu().float()
        want = _reference_real_call(emb, mask, srcs).float()
        assert torch.allclose(got, want), (
            f"patched injection wrong on MPS for the real call layout ({dtype})"
        )
