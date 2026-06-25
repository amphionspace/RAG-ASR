#!/usr/bin/env bash
# Pull the NVIDIA Triton image with retry logic, then extract /opt/tritonserver.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${RAG_ASR_TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.10-py3}"
RETRY_SLEEP_SECONDS="${RAG_ASR_DOCKER_PULL_RETRY_SLEEP_SECONDS:-60}"
MAX_ATTEMPTS="${RAG_ASR_DOCKER_PULL_MAX_ATTEMPTS:-0}"
TRITONSERVER_ROOT="${RAG_ASR_TRITONSERVER_ROOT:-$ROOT/var/tritonserver}"
EXTRACT_AFTER_PULL="${RAG_ASR_TRITON_EXTRACT_AFTER_PULL:-1}"
FORCE_EXTRACT="${RAG_ASR_TRITON_EXTRACT_FORCE:-0}"
BUILD_SERVER_ENV="${RAG_ASR_BUILD_TRITON_SERVER_ENV_AFTER_EXTRACT:-0}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

resolve_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not available on PATH." >&2
    exit 1
  fi

  if docker version >/dev/null 2>&1; then
    echo "docker"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    echo "sudo docker"
    return
  fi

  echo "ERROR: docker is available, but this user cannot access the Docker daemon and sudo is unavailable." >&2
  exit 1
}

DOCKER_CMD="$(resolve_docker)"
attempt=1

has_valid_tritonserver() {
  [[ -x "$TRITONSERVER_ROOT/bin/tritonserver" && -d "$TRITONSERVER_ROOT/backends/python" ]]
}

copy_extra_runtime_libraries() {
  local container_id="$1"
  local dst_dir="$2/lib"
  local copied=0
  local candidate

  mkdir -p "$dst_dir" || return 1

  # Some NGC releases keep these runtime libraries outside /opt/tritonserver.
  # Host-mode extraction needs local copies because start_triton.sh only adds
  # the extracted Triton lib directory to LD_LIBRARY_PATH.
  for candidate in \
    /lib/x86_64-linux-gnu/libb64.so.0d \
    /lib/x86_64-linux-gnu/libdcgm.so \
    /lib/x86_64-linux-gnu/libdcgm.so.3 \
    /lib/x86_64-linux-gnu/libdcgm.so.3.2.6 \
    /lib/x86_64-linux-gnu/libdcgm.so.3.3.6 \
    /usr/lib/x86_64-linux-gnu/libb64.so.0d \
    /usr/lib/x86_64-linux-gnu/libdcgm.so \
    /usr/lib/x86_64-linux-gnu/libdcgm.so.3 \
    /usr/lib/x86_64-linux-gnu/libdcgm.so.3.2.6 \
    /usr/lib/x86_64-linux-gnu/libdcgm.so.3.3.6; do
    if $DOCKER_CMD cp "$container_id:$candidate" "$dst_dir/" >/dev/null 2>&1; then
      copied=$((copied + 1))
    fi
  done

  if [[ "$copied" -gt 0 ]]; then
    log "Copied $copied extra runtime library file(s) into: $dst_dir"
  fi
}

verify_tritonserver_runtime() {
  local root_dir="$1"
  local ldd_output

  ldd_output="$(
    LD_LIBRARY_PATH="$root_dir/lib:${LD_LIBRARY_PATH:-}" \
      ldd "$root_dir/bin/tritonserver" 2>&1 || true
  )"

  if [[ "$ldd_output" == *"not found"* ]]; then
    log "ERROR: extracted tritonserver has unresolved runtime dependencies:"
    printf '%s\n' "$ldd_output" | python3 -c 'import sys
for line in sys.stdin:
    if "not found" in line:
        print(line.rstrip())'
    if [[ "$ldd_output" == *"GLIBC_"* ]]; then
      log "The selected image is newer than this host glibc. Use an Ubuntu-22.04-based Triton image such as nvcr.io/nvidia/tritonserver:24.10-py3, or run Triton inside Docker."
    fi
    return 1
  fi
}

extract_tritonserver() {
  if [[ "$EXTRACT_AFTER_PULL" != "1" ]]; then
    log "Skipping Triton extraction because RAG_ASR_TRITON_EXTRACT_AFTER_PULL=$EXTRACT_AFTER_PULL"
    return 0
  fi

  if has_valid_tritonserver && [[ "$FORCE_EXTRACT" != "1" ]]; then
    if verify_tritonserver_runtime "$TRITONSERVER_ROOT"; then
      log "Triton server already exists: $TRITONSERVER_ROOT"
      log "Set RAG_ASR_TRITON_EXTRACT_FORCE=1 to replace it."
      return 0
    fi

    log "Existing Triton server is not compatible with this host: $TRITONSERVER_ROOT"
    log "Set RAG_ASR_TRITON_EXTRACT_FORCE=1 to move it aside and extract again."
    return 1
  fi

  if [[ -e "$TRITONSERVER_ROOT" && "$FORCE_EXTRACT" != "1" ]]; then
    log "ERROR: destination exists but is not a complete Triton server: $TRITONSERVER_ROOT"
    log "Set RAG_ASR_TRITON_EXTRACT_FORCE=1 to move it aside and extract again."
    return 1
  fi

  local parent_dir
  local tmp_dir
  local container_id
  parent_dir="$(dirname "$TRITONSERVER_ROOT")"
  tmp_dir="${TRITONSERVER_ROOT}.tmp.$$"
  container_id=""

  log "Extracting /opt/tritonserver from image into: $TRITONSERVER_ROOT"
  mkdir -p "$parent_dir" || return 1

  if [[ -e "$tmp_dir" ]]; then
    log "ERROR: temporary extraction path already exists: $tmp_dir"
    return 1
  fi
  mkdir -p "$tmp_dir" || return 1

  container_id="$($DOCKER_CMD create "$IMAGE")"
  if [[ -z "$container_id" ]]; then
    log "ERROR: failed to create temporary container from $IMAGE"
    rm -rf "$tmp_dir"
    return 1
  fi

  if ! $DOCKER_CMD cp "$container_id":/opt/tritonserver/. "$tmp_dir/"; then
    log "ERROR: failed to copy /opt/tritonserver from container $container_id"
    $DOCKER_CMD rm "$container_id" >/dev/null 2>&1 || true
    rm -rf "$tmp_dir"
    return 1
  fi

  copy_extra_runtime_libraries "$container_id" "$tmp_dir" || {
    $DOCKER_CMD rm "$container_id" >/dev/null 2>&1 || true
    rm -rf "$tmp_dir"
    return 1
  }

  $DOCKER_CMD rm "$container_id" >/dev/null 2>&1 || true

  if [[ ! -x "$tmp_dir/bin/tritonserver" || ! -d "$tmp_dir/backends/python" ]]; then
    log "ERROR: extracted Triton server is incomplete: $tmp_dir"
    rm -rf "$tmp_dir"
    return 1
  fi

  if ! verify_tritonserver_runtime "$tmp_dir"; then
    rm -rf "$tmp_dir"
    return 1
  fi

  if [[ -e "$TRITONSERVER_ROOT" ]]; then
    local backup_dir
    backup_dir="${TRITONSERVER_ROOT}.bak.$(date +%Y%m%d%H%M%S)"
    log "Moving existing Triton server aside: $backup_dir"
    mv "$TRITONSERVER_ROOT" "$backup_dir" || {
      rm -rf "$tmp_dir"
      return 1
    }
  fi

  mv "$tmp_dir" "$TRITONSERVER_ROOT" || return 1
  log "Extraction succeeded: $TRITONSERVER_ROOT"
}

build_server_env() {
  if [[ "$BUILD_SERVER_ENV" != "1" ]]; then
    log "Skipping server env build. To run it automatically, set RAG_ASR_BUILD_TRITON_SERVER_ENV_AFTER_EXTRACT=1."
    log "Manual command:"
    log "  RAG_ASR_TRITONSERVER_ROOT=\"$TRITONSERVER_ROOT\" bash scripts/build_triton_server_env.sh"
    return 0
  fi

  log "Building Triton server conda env with: $TRITONSERVER_ROOT"
  RAG_ASR_TRITONSERVER_ROOT="$TRITONSERVER_ROOT" bash "$ROOT/scripts/build_triton_server_env.sh"
}

log "Pulling Triton image: $IMAGE"
log "Docker command: $DOCKER_CMD"
if [[ "$MAX_ATTEMPTS" == "0" ]]; then
  log "Max attempts: unlimited"
else
  log "Max attempts: $MAX_ATTEMPTS"
fi
log "Retry sleep: ${RETRY_SLEEP_SECONDS}s"

while true; do
  log "Attempt $attempt started."

  # Docker reuses completed layers, so retrying after a network reset usually
  # continues from the layers that already finished successfully.
  if $DOCKER_CMD pull "$IMAGE"; then
    log "Pull succeeded: $IMAGE"
    extract_tritonserver || exit 1
    build_server_env || exit 1
    exit 0
  fi

  log "Attempt $attempt failed."
  if [[ "$MAX_ATTEMPTS" != "0" && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
    log "Giving up after $attempt attempts."
    exit 1
  fi

  attempt=$((attempt + 1))
  log "Sleeping ${RETRY_SLEEP_SECONDS}s before retry."
  sleep "$RETRY_SLEEP_SECONDS"
done
