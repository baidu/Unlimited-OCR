# R-SWA + 翻译示例：Unlimited-OCR → R-SWA Qwen2.5

本示例演示 **R-SWA（Reference Sliding Window Attention）的通用性**——
百度 Unlimited-OCR 用 R-SWA 一次性解析长文档，本示例把它的输出（带版式 markdown）
再交给一个用 **R-SWA 增强的 Qwen2.5 翻译模型**，完成「文档解析 → 翻译」长链路。

> 这正好呼应 Unlimited-OCR 论文的核心判断：**R-SWA 是通用解析机制，OCR 只是第一站。**
> 翻译是 R-SWA 在 OCR 之外的第二个落点：解析侧与翻译侧都用「源文 pin 住 + 生成段环形滑窗」，
> 长链路两端 KV Cache 都恒定有界。

## 流水线

```
图片/PDF ──[Unlimited-OCR (R-SWA)]──> 带版式文本
                                     │
                                     ▼ 分段（按 <det> 区块）
                          [R-SWA Qwen2.5 翻译] 逐段翻译
                                     │
                                     ▼ 拼回
                               带版式译文
```

## 为什么用 R-SWA 做翻译

| 机制 | 译文 KV 长度 | 长文档 | 跨段一致性 |
|------|-------------|--------|-----------|
| 普通全注意力 | 随译文线性增长 → 爆显存 | ❌ | ✅（但撑不住） |
| 普通滑窗 | 有界 | ✅ | ❌（丢源文上下文） |
| **R-SWA（本示例）** | **恒 = prefill_len + W** | ✅ | ✅（源文 pin 住） |

R-SWA 把**源文/prefill 永久 pin 在 KV 头部**，只有生成的译文走环形滑窗 W。
因此翻译长文档时：显存不随译文变长，且每一段都能看到完整源文，跨段术语/指代一致。

## 实测（Qwen2.5-0.5B, RTX 4060, real GPU）

生成 384 token 的 KV Cache 与吞吐（prefill_len=117）：

| 窗口 W | 稳态 KV 长度 | KV 体积 | 解码速度 | 相对全注意力省 KV |
|--------|-------------|---------|---------|------------------|
| 全注意力（对照） | 501 | 12.3 MB | 28.9 tok/s | — |
| W=256 | 373 | 9.17 MB | 27.0 tok/s | 25.6% |
| W=128 | 245 | 6.02 MB | 26.5 tok/s | 51.1% |
| W=64  | 181 | 4.45 MB | 25.7 tok/s | 63.9% |

投影到更长生成（1024 token）：W=64/128/256 分别省 **84.1% / 78.5% / 67.3%** KV，
且精度（与全注意力 cosine 相似度 ≈ 1.0）几乎无损。

## 运行

```bash
cd examples/rswa_translate
pip install -r requirements.txt

# 默认从 Hugging Face Hub 拉取 baidu/Unlimited-OCR 与 Qwen/Qwen2.5-0.5B
python ocr_translate.py path/to/your/image.png --output_dir outputs

# 离线 / 本地权重：用环境变量指到本地缓存或权重
export UOCR_MODEL=baidu/Unlimited-OCR          # 已缓存时可离线
export RSWA_QWEN25=/path/to/Qwen2.5-0.5B       # 本地翻译模型
export HF_HOME=/path/to/hf_cache               # 指向本地 HF 缓存
python ocr_translate.py image.png --output_dir outputs
```

输出：`outputs/source.md`（OCR 原文）、`outputs/translated.md`（R-SWA 译文）。

## 文件

- `rswa_qwen25.py` — Qwen2/Qwen2.5 的 R-SWA 注意力实现（源自 Unlimited-OCR 的
  `SlidingWindowLlamaAttention` 思想，搬到 Qwen2 架构；权重零改动，仅替换注意力层）。
- `ocr_translate.py` — 端到端流水线：OCR → 分段 → R-SWA 翻译 → 拼回。
- `requirements.txt` — 依赖。

## 依赖

- `transformers>=4.57.1`（DynamicCache API 对齐）
- `torch`（CUDA 12.x 推荐）
- 本示例在 `python 3.12` + CUDA 12.8 验证通过。
