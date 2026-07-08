# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/tritonserver:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    RAG_ASR_HOME=/opt/rag-asr \
    RAG_ASR_CONFIG=/opt/rag-asr/configs/serve.compose.yaml \
    RAG_ASR_CONTAINER_PYTHON=/opt/rag-asr/triton-exec/bin/python \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    PATH=/opt/rag-asr/triton-exec/bin:/usr/local/bin:${PATH} \
    PYTHONNOUSERSITE=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl aria2 bzip2 python3.10-venv && \
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
    --mount=type=cache,target=/tmp/rag-asr-wheels \
    set -e; \
    WHEEL_DIR=/tmp/rag-asr-wheels; \
    mkdir -p "$WHEEL_DIR"; \
    download_wheel() { \
      url="$1"; \
      out="$2"; \
      aria2c \
        -c -x16 -s16 -k1M \
        --retry-wait=3 --max-tries=0 \
        --timeout=30 --connect-timeout=30 \
        --summary-interval=30 \
        --dir="$WHEEL_DIR" \
        --out="$out" \
        "$url"; \
    }; \
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
    download_wheel \
      https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl \
      torch-2.10.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl; \
    download_wheel \
      https://download.pytorch.org/whl/cu128/torchaudio-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl \
      torchaudio-2.10.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl; \
    download_wheel \
      https://pypi.nvidia.com/nvidia-nvshmem-cu12/nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
      nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl; \
    download_wheel \
      https://pypi.nvidia.com/nvidia-nccl-cu12/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
      nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl; \
    download_wheel \
      https://files.pythonhosted.org/packages/8c/f7/f1c9d3424ab199ac53c2da567b859bcddbb9c9e7154805119f8bd95ec36f/triton-3.6.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
      triton-3.6.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl; \
    env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install \
      --no-deps \
      "$WHEEL_DIR/torch-2.10.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl" \
      "$WHEEL_DIR/torchaudio-2.10.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl" \
      "$WHEEL_DIR/nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
      "$WHEEL_DIR/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
      "$WHEEL_DIR/triton-3.6.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"; \
    pip_install_retry \
      --retries 10 --timeout 120 --resume-retries 10 \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -c /tmp/constraints-runtime.txt \
      transformers huggingface_hub "numpy<2" soundfile librosa faiss-cpu \
      fastapi uvicorn python-multipart PyYAML "tritonclient[http]" || exit 1; \
    pip_install_retry \
      --retries 10 --timeout 120 --resume-retries 10 \
      audioread click cytoolz intervaltree tabulate lilcom \
      cuda-bindings==12.9.4 jinja2 networkx sympy || exit 1; \
    env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install \
      --no-deps \
      -c /tmp/constraints-runtime.txt \
      lhotse || exit 1; \
    env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -c \
      "import torch, torchaudio, transformers, librosa, fastapi, uvicorn, soundfile, lhotse, triton; from transformers import AutoModelForCausalLM; assert torch.__version__ == '2.10.0+cu128'; assert torchaudio.__version__ == '2.10.0+cu128'; assert hasattr(triton, 'language')" || exit 1

COPY . /opt/rag-asr

RUN env -u PYTHONPATH /opt/rag-asr/triton-exec/bin/python -m pip install --no-cache-dir /opt/rag-asr --no-deps && \
    /opt/rag-asr/triton-exec/bin/venv-pack \
      -p /opt/rag-asr/triton-exec \
      -o /opt/rag-asr/triton-exec-env.tar.gz \
      --force && \
    mkdir -p /data/rag-asr/hotwords /data/rag-asr/retrieve_cache /data/rag-asr/enrollments /data/rag-asr/triton_repo && \
    chmod +x /opt/rag-asr/scripts/start_triton_container.sh

EXPOSE 10001 10002 10003 8080

WORKDIR /tmp

CMD ["/opt/rag-asr/scripts/start_triton_container.sh"]
