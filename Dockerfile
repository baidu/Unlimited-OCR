# Unlimited-OCR — Docker image
#
# Build:
#   docker build -t unlimited-ocr .
#
# Run (SGLang server on port 10000):
#   docker run --gpus all -p 10000:10000 unlimited-ocr
#
# Run with a local Hugging Face model directory:
#   docker run --gpus all -p 10000:10000 \
#     -v /path/to/model:/model \
#     unlimited-ocr --model /model
#
# Run batch inference with infer.py (launches server automatically):
#   docker run --gpus all -v /path/to/images:/data unlimited-ocr \
#     python infer.py --image-dir /data

FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the custom SGLang wheel first (rarely changes) so Docker can cache it
COPY wheel/ wheel/

# Install uv and set up the virtualenv
RUN pip3 install --no-cache-dir uv && \
    uv venv --python 3.12 .venv && \
    .venv/bin/pip install --no-cache-dir \
        wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl

# Copy and install the rest of the dependencies
COPY requirements.txt .
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code last (changes most often)
COPY infer.py .
COPY assets/ assets/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 10000

CMD ["python", "-m", "sglang.launch_server", \
    "--model", "baidu/Unlimited-OCR", \
    "--served-model-name", "Unlimited-OCR", \
    "--attention-backend", "fa3", \
    "--page-size", "1", \
    "--mem-fraction-static", "0.8", \
    "--context-length", "32768", \
    "--enable-custom-logit-processor", \
    "--disable-overlap-schedule", \
    "--skip-server-warmup", \
    "--host", "0.0.0.0", \
    "--port", "10000"]
