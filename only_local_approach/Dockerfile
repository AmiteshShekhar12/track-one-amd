FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
       build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# GGML_NATIVE=OFF -> portable amd64 binary (judging VM CPU may differ from
# the build machine). Build from source: the CPU wheel index's linux wheel
# is musl-linked and crashes on this glibc image.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.19

# Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB) -> compressed image ~5.2 GB, well
# under the 10 GB limit, and much stronger than a 3B on code/math/logic.
ARG MODEL_URL=https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
RUN mkdir -p /app/models \
    && curl -fL --http1.1 --retry 5 --retry-all-errors -C - \
       -o /app/models/model.gguf "$MODEL_URL"

COPY main.py .

ENV LOCAL_MODEL_PATH=/app/models/model.gguf

CMD ["python", "main.py"]
