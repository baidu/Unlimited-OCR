# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
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

RUN python3.12 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements-sglang.txt ./
COPY wheel/ ./wheel/
RUN python -m pip install -r requirements-sglang.txt

FROM ${CUDA_IMAGE} AS runtime

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
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

WORKDIR /app

COPY infer.py README.md LICENSE ./

RUN groupadd --gid "${GROUP_ID}" unlimited \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash unlimited \
    && mkdir -p /app/log /app/outputs "${HF_HOME}" \
    && chown -R unlimited:unlimited /app /home/unlimited

USER unlimited

EXPOSE 10000
VOLUME ["/data", "/app/outputs", "/app/log", "/home/unlimited/.cache/huggingface"]

ENTRYPOINT ["python", "infer.py"]
CMD ["--help"]
