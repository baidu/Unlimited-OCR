"""
rswa_qwen25.py — R-SWA (Reference Sliding Window Attention) 移植到 Qwen2 / Qwen2.5

移植自 baidu/Unlimited-OCR 的 `SlidingWindowLlamaAttention`，把"参考段 pin 住 +
生成段环形滑窗"机制搬到 Qwen2 架构上。

机制（KV Cache 寻址，仅此一层在动，权重零改动）：

    [0, prefill_len)           源文 / prompt  —— 永久 pin，永不覆盖
    [prefill_len, prefill_len+W)  已生成译文   —— 环形缓冲区，第 t 步写 (prefill_len + t%W)

W 在 forward 时从 `model.config._ring_window` 读取，是**运行时旋钮**而非训练常量：
    W = None  -> 全注意力（对照臂）
    W = 128/512/2048 -> 不同滑窗大小（消融臂）

prefill 路径改用 `F.scaled_dot_product_attention`，避免原 eager 实现在长序列上
完整物化注意力矩阵导致的 OOM（见 ANALYSIS-001 §4）。

用法：
    from rswa_qwen25 import patch_rswa
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
                                                 attn_implementation="eager")
    patch_rswa(model, window=512)          # R-SWA 实验臂
    # 或 patch_rswa(model, window=None)     # 全注意力对照臂
    model.generate(...)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class SlidingWindowQwen2Attention(Qwen2Attention):
    """Qwen2Attention + 仅对生成段生效的环形滑窗 KV Cache。

    源文/prefill 区永久 pin，只有 decode 侧的生成历史被限制在窗口 W 内。
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # 确保持有 rotary_emb（新版本 transformers 已上移到 model 级）
        if not hasattr(self, "rotary_emb") or self.rotary_emb is None:
            self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        # 暂存滑动窗口，便于在 config 上中和它，防止 DynamicCache 截断 pin 区
        self._sliding_window = getattr(config, "sliding_window", None)

    # ---- cache 访问垫片（兼容 transformers 版本漂移）----
    @staticmethod
    def _kcache(cache, layer_idx):
        return (
            cache.key_cache[layer_idx]
            if hasattr(cache, "key_cache")
            else cache.layers[layer_idx].keys
        )

    @staticmethod
    def _vcache(cache, layer_idx):
        return (
            cache.value_cache[layer_idx]
            if hasattr(cache, "value_cache")
            else cache.layers[layer_idx].values
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        if past_key_value is None:
            past_key_value = kwargs.get("past_key_values", None)

        bsz, q_len, _ = hidden_states.size()
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim
        num_kv_groups = self.num_key_value_groups

        W = getattr(self.config, "_ring_window", None)

        # ---------------- 共享 QKV 投影 + RoPE ----------------
        def project():
            q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            return q, k, v

        # ---------------- prefill / 全注意力路径 ----------------
        # 真 prefill：W 未设、无 cache、或首次前向（prefill 长度未知）
        is_prefill = (
            W is None
            or past_key_value is None
            or (
                q_len > 1
                and (
                    not hasattr(past_key_value, "_prefill_length")
                    or self.layer_idx not in past_key_value._prefill_length
                )
            )
        )

        if is_prefill:
            q, k, v = project()
            if past_key_value is not None:
                k, v = past_key_value.update(k, v, self.layer_idx)
            k_e = repeat_kv(k, num_kv_groups)
            v_e = repeat_kv(v, num_kv_groups)
            # SDPA：修复长 prefill 的 eager OOM
            attn_mask = (
                attention_mask
                if (attention_mask is not None and attention_mask.ndim == 4)
                else None
            )
            out = F.scaled_dot_product_attention(
                q, k_e, v_e, attn_mask=attn_mask, dropout_p=0.0, is_causal=(attn_mask is None)
            )
            out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            out = self.o_proj(out)
            if W is not None and past_key_value is not None and q_len > 1:
                if not hasattr(past_key_value, "_prefill_length"):
                    past_key_value._prefill_length = {}
                if self.layer_idx not in past_key_value._prefill_length:
                    past_key_value._prefill_length[self.layer_idx] = self._kcache(
                        past_key_value, self.layer_idx
                    ).shape[-2]
            return out, None

        # ---------------- decode / 环形路径 ----------------
        pk = past_key_value
        if not hasattr(pk, "_prefill_length"):
            pk._prefill_length = {}
        if self.layer_idx not in pk._prefill_length:
            pk._prefill_length[self.layer_idx] = self._kcache(pk, self.layer_idx).shape[-2]
        prefill_len = pk._prefill_length[self.layer_idx]
        cur_len = self._kcache(pk, self.layer_idx).shape[-2]

        # 预热：追加直到环形区填满，再切入稳态覆盖
        if cur_len < prefill_len + W:
            q, k, v = project()
            if pk is not None:
                k, v = pk.update(k, v, self.layer_idx)
            k_e = repeat_kv(k, num_kv_groups)
            v_e = repeat_kv(v, num_kv_groups)
            attn_mask = (
                attention_mask
                if (attention_mask is not None and attention_mask.ndim == 4)
                else None
            )
            out = F.scaled_dot_product_attention(
                q, k_e, v_e, attn_mask=attn_mask, dropout_p=0.0, is_causal=(attn_mask is None)
            )
            out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            out = self.o_proj(out)
            new_len = self._kcache(pk, self.layer_idx).shape[-2]
            if new_len >= prefill_len + W:
                if not hasattr(pk, "_ring_pos"):
                    pk._ring_pos = {}
                pk._ring_pos[self.layer_idx] = 0
            return out, None

        # 稳态：环形就地覆盖
        if not hasattr(pk, "_ring_pos") or self.layer_idx not in pk._ring_pos:
            pk._ring_pos = getattr(pk, "_ring_pos", {}) or {}
            pk._ring_pos[self.layer_idx] = 0
        ring_pos = pk._ring_pos[self.layer_idx]

        q, k, v = project()
        kcache = self._kcache(pk, self.layer_idx)
        vcache = self._vcache(pk, self.layer_idx)
        for t in range(q_len):
            slot = prefill_len + ring_pos
            kcache[:, :, slot : slot + 1, :] = k[:, :, t : t + 1, :]
            vcache[:, :, slot : slot + 1, :] = v[:, :, t : t + 1, :]
            ring_pos = (ring_pos + 1) % W
        pk._ring_pos[self.layer_idx] = ring_pos

        k_e = repeat_kv(kcache, num_kv_groups)
        v_e = repeat_kv(vcache, num_kv_groups)
        # decode 时 q_len == 1，矩阵极小，无需 causal mask
        attn = torch.matmul(q, k_e.transpose(2, 3)) / math.sqrt(head_dim)
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_e)
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = self.o_proj(out)
        return out, None


def patch_rswa(model, window: int | None = 512):
    """把 `model` 里所有 Qwen2Attention 替换为 SlidingWindowQwen2Attention，
    并通过 `model.config._ring_window` 路由窗口大小。

    window=None -> 全注意力（对照臂）
    window=int  -> R-SWA（实验臂 / 消融臂）
    """
    model.config._ring_window = window
    # 中和 config 中的 sliding_window，避免 DynamicCache 截断 pin 区
    model.config.sliding_window = None

    for layer in model.model.layers:
        idx = layer.self_attn.layer_idx
        ref = layer.self_attn
        new_attn = SlidingWindowQwen2Attention(model.config, idx)
        new_attn.load_state_dict(ref.state_dict(), strict=False)
        # 对齐 device + dtype：新建层默认 float32 / 留在 CPU，模型可能已上 cuda+fp16
        # → 只转 dtype 会让新模块停在 CPU，引发 "mat1 cuda / 其他 cpu" 设备不匹配。
        # 同时对齐 device（取被替换层的真实 device），不再依赖调用方补 model.to(device)。
        _dev = ref.q_proj.weight.device
        _dt = ref.q_proj.weight.dtype
        new_attn = new_attn.to(device=_dev, dtype=_dt)
        layer.self_attn = new_attn
    return model


if __name__ == "__main__":
    # 冒烟测试：证明 R-SWA 能跑通 generate 且不崩溃。需 torch + 权重。
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = os.environ.get("RSWA_MODEL", "Qwen/Qwen2.5-0.5B")
    print(f"[rswa] loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    patch_rswa(model, window=512)
    model.eval()

    src = "Translate to Chinese: The sliding window keeps only the most recent tokens."
    ids = tok(src, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=64)
    print("[rswa] output:", tok.decode(out[0], skip_special_tokens=True))
    print("[rswa] OK — R-SWA forward+generate ran without error.")
