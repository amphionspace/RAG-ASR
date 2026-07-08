# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/tritonserver:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    RAG_ASR_HOME=/opt/rag-asr \
    RAG_ASR_CONFIG=/opt/rag-asr/configs/serve.compose.yaml \
    RAG_ASR_CONTAINER_PYTHON=/opt/rag-asr/triton-exec/bin/python \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/rag-asr/triton-exec/bin:/usr/local/bin:${PATH} \
    PYTHONNOUSERSITE=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl bzip2 python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --copies /opt/rag-asr/triton-exec && \
    env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install \
      --retries 10 --timeout 120 \
      --upgrade pip setuptools wheel && \
    env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install \
      --retries 10 --timeout 120 \
      venv-pack==0.2.0

WORKDIR /opt/rag-asr

COPY docker/constraints-runtime.txt /tmp/constraints-runtime.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    set -e; \
    pip_install_retry() { \
      for attempt in 1 2 3 4 5; do \
        if env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install "$@"; then \
          return 0; \
        fi; \
        status="$?"; \
        if [ "$attempt" = "5" ]; then \
          return "$status"; \
        fi; \
        sleep "$((attempt * 20))"; \
      done; \
    }; \
    pip_install_retry \
      --retries 10 --timeout 120 --resume-retries 10 \
      torch==2.10.0+cu128 torchaudio==2.10.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128 || exit 1; \
    env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -c \
      "import torch, torchaudio; assert torch.__version__ == '2.10.0+cu128'; assert torchaudio.__version__ == '2.10.0+cu128'" || exit 1; \
    pip_install_retry \
      --retries 10 --timeout 120 --resume-retries 10 \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -c /tmp/constraints-runtime.txt \
      transformers huggingface_hub lhotse "numpy<2" soundfile librosa faiss-cpu \
      fastapi uvicorn python-multipart PyYAML "tritonclient[http]" || exit 1

COPY . /opt/rag-asr

RUN env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install --no-cache-dir /opt/rag-asr --no-deps && \
    /opt/rag-asr/triton-exec/bin/venv-pack \
      -p /opt/rag-asr/triton-exec \
      -o /opt/rag-asr/triton-exec-env.tar.gz \
      --force && \
    mkdir -p /data/rag-asr/hotwords /data/rag-asr/retrieve_cache /data/rag-asr/enrollments /data/rag-asr/triton_repo && \
    chmod +x /opt/rag-asr/scripts/start_triton_container.sh

EXPOSE 10001 10002 10003 8080

CMD ["/opt/rag-asr/scripts/start_triton_container.sh"]
