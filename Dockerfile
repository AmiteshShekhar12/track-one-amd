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
    CMAKE_BUILD_PARALLEL_LEVEL=4

# llama-cpp-python is pinned to a version with a prebuilt cp312 linux_x86_64
# wheel on the CPU index, so the cross-build normally installs in seconds;
# the toolchain above is fallback insurance if the wheel index is unavailable.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python==0.3.19 \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Bundle local model weights directly in the image — the judging VM has no
# Ollama or model runtime pre-installed. Override MODEL_URL to swap models.
ARG MODEL_URL=https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf
RUN mkdir -p /app/models \
    && curl -fL --http1.1 --retry 5 --retry-all-errors -C - \
       -o /app/models/model.gguf "$MODEL_URL"

COPY main.py .

ENV LOCAL_MODEL_PATH=/app/models/model.gguf

CMD ["python", "main.py"]
