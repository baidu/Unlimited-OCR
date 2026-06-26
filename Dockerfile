# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-sglang.txt ./
COPY wheel/ ./wheel/
RUN python3.12 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && mkdir /wheelhouse \
    && python -m pip wheel --wheel-dir /wheelhouse -r requirements-sglang.txt

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV HF_HOME=/home/unlimited/.cache/huggingface \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheelhouse /wheelhouse
RUN python3.12 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-index --find-links=/wheelhouse \
        "sglang==0.0.0.dev11416+g92e8bb79e" \
        "pymupdf==1.27.2.2" \
    && rm -rf /wheelhouse

WORKDIR /app

COPY infer.py README.md LICENSE ./

RUN groupadd --gid "${GROUP_ID}" unlimited \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash unlimited \
    && mkdir -p /app/log /app/outputs "${HF_HOME}" \
    && chown -R unlimited:unlimited /app /home/unlimited

USER unlimited

EXPOSE 10000
VOLUME ["/data", "/app/outputs", "/app/log", "/home/unlimited/.cache/huggingface"]

CMD ["python", "-m", "sglang.launch_server", \
    "--model", "baidu/Unlimited-OCR", \
    "--trust-remote-code", \
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
