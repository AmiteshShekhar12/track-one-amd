FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# GGML_NATIVE=OFF: a source build must not tune for the build machine's CPU —
# under buildx/QEMU that's an emulated CPU, and the judging VM's amd64 CPU may
# not support the same instruction extensions (illegal-instruction crash).
# CMAKE_BUILD_PARALLEL_LEVEL caps compile jobs: full-parallel C++ under QEMU
# can OOM the Docker Desktop VM.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# llama-cpp-python has no manylinux wheel on PyPI for this version (sdist
# only), and the "CPU wheel index" alternative (abetlen.github.io) ships a
# plain `linux_x86_64`-tagged wheel that is actually musl-linked — it
# installs cleanly but crashes at import time on this glibc-based Debian
# image ("libc.musl-x86_64.so.1: cannot open shared object file"), silently
# disabling the entire local-model cost-saving path. Always build from
# source here so the extension links against the image's real glibc; the
# toolchain above (build-essential/cmake) plus GGML_NATIVE=OFF makes this a
# few-minutes compile, not an hours-long one.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.19

# Bundle local model weights directly in the image — the judging VM has no
# Ollama or model runtime pre-installed. Override MODEL_URL to swap models.
ARG MODEL_URL=https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf
RUN mkdir -p /app/models \
    && curl -fL --http1.1 --retry 5 --retry-all-errors -C - \
       -o /app/models/model.gguf "$MODEL_URL"

COPY main.py .

ENV LOCAL_MODEL_PATH=/app/models/model.gguf

CMD ["python", "main.py"]
