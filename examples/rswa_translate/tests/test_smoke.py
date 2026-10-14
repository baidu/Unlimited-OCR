"""重路径冒烟测试：需要 GPU + 真实权重，默认 CI 跳过。

设环境变量 ``RSWA_RUN_HEAVY=1`` 可本地开启；用于验证 R-SWA 能真正
patch 进 Qwen2.5 并把窗口参数写进 config（论文中的"权重零改动"假设）。
"""

import os

import pytest

HEAVY = pytest.mark.skipif(
    os.environ.get("RSWA_RUN_HEAVY") != "1",
    reason="重路径需 CUDA + Qwen2.5/Unlimited-OCR 权重；设 RSWA_RUN_HEAVY=1 开启",
)


@HEAVY
def test_patch_rswa_writes_window_to_config():
    from rswa_qwen25 import patch_rswa
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
    patch_rswa(model, window=64)
    # window=None 退化为全注意力（对照臂）；int 即 R-SWA 实验臂。
    assert model.config._ring_window == 64
    assert model.config.sliding_window is None


@HEAVY
def test_patch_rswa_full_attention_arm():
    from rswa_qwen25 import patch_rswa
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
    patch_rswa(model, window=None)
    assert model.config._ring_window is None
