# Unlimited OCR · Mac (Apple Silicon) 适配版

> **[English](#english)** | **中文**

---

## 中文

在 **Mac Apple Silicon (M1 / M2 / M3 / M4)** 上本地跑通 [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (6.7B VLM) — 带 grounding 的 OCR,直接输出 `<|det|><类型> [bbox] <|/det|><文字>` 格式。

**官方只支持 NVIDIA CUDA 12.9+**。这个仓库是社区维护的 Mac 适配:打 3 个 patch,其它都和上游一致。

### 实测性能 (M4 Pro 48GB)

| 任务 | 耗时 |
|---|---|
| 模型加载(从 SSD 读 6.7GB) | ~3s |
| 简单图 (768×1024) | ~9s |
| 复杂文档 (1024×1024, 13 行) | ~19s |
| 多图批处理 | ~20s/图 |

### 一键安装

```bash
git clone https://github.com/Konsn666/unlimited-ocr-mac.git
cd unlimited-ocr-mac
bash scripts/install.sh          # ~5-10 分钟(下载 6.7GB 模型)
```

模型默认装到 `./model_dir`,缓存走 `~/.cache/huggingface/hub/`。装外置 SSD:

```bash
HF_HUB_CACHE="/Volumes/外置盘/Model/hub" bash scripts/install.sh
```

### 一键运行

```bash
# 单图
bash scripts/run.sh /path/to/your.png

# 多图批处理
bash scripts/run.sh --image_dir /path/to/images/ --output_dir ./out

# 高级参数
bash scripts/run.sh /path/to/image.png \
    --base_size 1024 --image_size 640 --max_length 8192 \
    --no_repeat_ngram_size 35 --ngram_window 128
```

### 作为 MCP 服务 (给 Claude Code / Codex / 其他 AI 工具调用)

```bash
bash scripts/mcp.sh    # 启动 stdio MCP server
```

注册到 Claude Code (`~/.claude.json` 或项目 `.mcp.json`):
```json
{
  "mcpServers": {
    "unlimited-ocr": {
      "command": "python3",
      "args": ["/path/to/unlimited-ocr-mac/mcp_server/server.py"]
    }
  }
}
```

注册到 OpenAI Codex CLI (`~/.codex/config.toml`):
```toml
[mcp_servers.unlimited-ocr]
command = "python3"
args = ["/abs/path/to/unlimited-ocr-mac/mcp_server/server.py"]
```

启动后,你的 AI 工具就能调用:
- `ocr_image(path)` — 单张图 OCR,返回 JSON
- `ocr_directory(path)` — 批处理目录
- `model_status()` — 看模型加载状态

### 三个 Mac 适配 patch (核心)

模型官方代码假设 NVIDIA GPU,在 Mac 上要改三处:

1. **`.cuda()` 硬编码 → `.to('mps')`** (`modeling_unlimitedocr.py` 里 13+ 处)
   - `run_mac.py::apply_patches()` 自动改写

2. **`torch.autocast("cuda", ...)` → 禁用** ⭐ **最关键**
   - PyTorch MPS backend 对 `torch.autocast(device_type="mps", dtype=bfloat16)` 有 bug,会让 MLA+MoE 算子精度漂移
   - 直接去掉 autocast,让模型用自然 bf16 推理
   - 表现:有 autocast → 5-10 token 后 logits 塌缩成 `168168168...` 死循环;去掉 → 完美工作

3. **`SlidingWindowNoRepeatNgramProcessor` 继承 `LogitsProcessor`**
   - 原版是个普通类,transformers 4.57 不会调用它
   - 加 `from transformers import LogitsProcessor` + `(LogitsProcessor)` 让 ngram 真正生效

这些 patch 都在 `patches/` 目录里,`install.sh` 自动覆盖到下载的 HF 模型。

### 输出格式

每行一个 detection:
```
<|det|><type> [x1, y1, x2, y2]<|/det|><text>
```

- `<type>`: `title` / `text` / `header` / `table` / `[Non-Text]` 等
- `[x1, y1, x2, y2]`: 4-corner bounding box(像素)
- `<text>`: 识别出的文本

示例输出:
```markdown
<|det|>title [35, 38, 685, 88]<|/det|>Unlimited OCR 真实文档测试
<|det|>text [33, 114, 761, 150]<|/det|>百度于2026年6月22日发布 Unlimited-OCR 模型。
<|det|>text [33, 150, 576, 179]<|/det|>模型基于 Deepseek V2 架构,采用 MoE 路由机制。
```

### 系统要求

- macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4)
- Python 3.12 或 3.13 (推荐 3.13)
- 统一内存 **≥ 24GB** (推荐 32GB+;48GB 实测无压力)
- 约 **8GB 磁盘** 给模型权重

### 已知限制

- **不官方支持**:社区 patch,不是 Baidu 官方 Mac 路径
- **流式输出禁用**:PyTorch MPS 有 "Placeholder storage" bug(2.12+);用 `eval_mode=True` 一次性返回代替
- **fp16 路径有 dtype mismatch**:推荐 bf16
- **速度** ~20s/图,比 NVIDIA GPU 慢 5-10x,比 PaddleOCR 等传统 OCR 准确度高

### 致谢

- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) - 原始仓库 (MIT)
- [huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) - 模型权重
- PyTorch MPS backend 团队

---

<a id="english"></a>

## English

Run [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (6.7B VLM) **locally on Mac Apple Silicon (M1/M2/M3/M4)**. Outputs grounded OCR as `<|det|><type> [bbox] <|/det|><text>`.

**Officially NVIDIA CUDA 12.9+ only.** This repo is a community-maintained Mac adaptation with 3 patches; everything else is identical to upstream.

### Benchmarks (M4 Pro 48GB)

| Task | Time |
|---|---|
| Model load (6.7GB from SSD) | ~3s |
| Simple image (768×1024) | ~9s |
| Complex doc (1024×1024, 13 lines) | ~19s |
| Batch processing | ~20s/image |

### One-click install

```bash
git clone https://github.com/Konsn666/unlimited-ocr-mac.git
cd unlimited-ocr-mac
bash scripts/install.sh          # ~5-10 min (downloads 6.7GB model)
```

Model installs to `./model_dir`; cache goes to `~/.cache/huggingface/hub/`. For an external SSD:

```bash
HF_HUB_CACHE="/Volumes/external/Model/hub" bash scripts/install.sh
```

### One-click run

```bash
# Single image
bash scripts/run.sh /path/to/your.png

# Batch
bash scripts/run.sh --image_dir /path/to/images/ --output_dir ./out

# Advanced
bash scripts/run.sh /path/to/image.png \
    --base_size 1024 --image_size 640 --max_length 8192 \
    --no_repeat_ngram_size 35 --ngram_window 128
```

### MCP service (for Claude Code / Codex / other AI tools)

```bash
bash scripts/mcp.sh    # start stdio MCP server
```

Register with Claude Code (`~/.claude.json` or project `.mcp.json`):
```json
{
  "mcpServers": {
    "unlimited-ocr": {
      "command": "python3",
      "args": ["/path/to/unlimited-ocr-mac/mcp_server/server.py"]
    }
  }
}
```

Register with OpenAI Codex CLI (`~/.codex/config.toml`):
```toml
[mcp_servers.unlimited-ocr]
command = "python3"
args = ["/abs/path/to/unlimited-ocr-mac/mcp_server/server.py"]
```

Once registered, your AI client can call:
- `ocr_image(path)` — single image OCR, returns JSON
- `ocr_directory(path)` — batch directory
- `model_status()` — check model load state

### Three Mac adaptation patches (key)

Upstream code assumes NVIDIA GPUs; on Mac you need to patch three things:

1. **`.cuda()` hardcoded → `.to('mps')`** (13+ places in `modeling_unlimitedocr.py`)
   - `run_mac.py::apply_patches()` rewrites these automatically

2. **Disable `torch.autocast("cuda", ...)`** ⭐ **most critical**
   - PyTorch's MPS backend has a bug with `torch.autocast(device_type="mps", dtype=bfloat16)` — it causes precision drift in MLA+MoE ops
   - Remove autocast; let the model use its natural bf16 dtype
   - Symptom with autocast: 5-10 tokens in, logits collapse into `168168168...` infinite loop. Without: works perfectly.

3. **`SlidingWindowNoRepeatNgramProcessor` inherit from `LogitsProcessor`**
   - Upstream is a plain class, transformers 4.57 won't call it
   - Add `from transformers import LogitsProcessor` + `(LogitsProcessor)` to make ngram actually apply

Patches live in `patches/`; `install.sh` overlays them on the downloaded HF model automatically.

### Output format

One line per detection:
```
<|det|><type> [x1, y1, x2, y2]<|/det|><text>
```

- `<type>`: `title` / `text` / `header` / `table` / `[Non-Text]` etc.
- `[x1, y1, x2, y2]`: 4-corner bounding box (pixels)
- `<text>`: recognized text

Example:
```markdown
<|det|>title [35, 38, 685, 88]<|/det|>Unlimited OCR Real Document Test
<|det|>text [33, 114, 761, 150]<|/det|>Baidu released Unlimited-OCR model on 2026-06-22.
<|det|>text [33, 150, 576, 179]<|/det|>Model is based on Deepseek V2 architecture with MoE routing.
```

### Requirements

- macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4)
- Python 3.12 or 3.13 (3.13 recommended)
- Unified memory **≥ 24GB** (32GB+ recommended; 48GB tested no problem)
- ~**8GB disk** for model weights

### Known limitations

- **Not officially supported** — community patches, not Baidu's official Mac path
- **Streaming output disabled** — PyTorch MPS has a "Placeholder storage" bug; we use `eval_mode=True` (returns full string at once) instead
- **fp16 path has dtype mismatch** — use bf16
- **Speed** ~20s/image, ~5-10× slower than NVIDIA GPU but more accurate than traditional OCR (PaddleOCR, etc.)

### Credits

- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) — original repo (MIT)
- [huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) — model weights
- PyTorch MPS backend team
