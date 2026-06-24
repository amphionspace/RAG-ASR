#!/usr/bin/env bash
# Show the current Triton RAG-ASR hotword service status.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

URL="${RAG_ASR_TRITON_URL:-localhost:8000}"
MODEL="${RAG_ASR_TRITON_MODEL:-rag_asr_retrieve}"
LIMIT="${RAG_ASR_STATUS_LIMIT:-10}"
OFFSET="${RAG_ASR_STATUS_OFFSET:-0}"
QUERY="${RAG_ASR_STATUS_QUERY:-}"
VERBOSE=0
FORMAT="text"
CONDA_ENV="${RAG_ASR_CLIENT_CONDA_ENV:-triton}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/hotword_status.sh [options]

Options:
  -u URL      Triton HTTP URL (default: localhost:8000)
  -m MODEL    Triton model name (default: rag_asr_retrieve)
  -l LIMIT    Number of hotwords to show (default: 10)
  -o OFFSET   Hotword list offset (default: 0)
  -q QUERY    Substring filter for hotwords
  -v          Include full Triton input/output schema
  -j          Print raw JSON instead of human-readable text
  -h          Show this help

Environment overrides:
  RAG_ASR_TRITON_URL
  RAG_ASR_TRITON_MODEL
  RAG_ASR_STATUS_LIMIT
  RAG_ASR_STATUS_OFFSET
  RAG_ASR_STATUS_QUERY
  RAG_ASR_CLIENT_CONDA_ENV
  RAG_ASR_SKIP_CONDA=1
EOF
}

while getopts ":u:m:l:o:q:vjh" opt; do
  case "$opt" in
    u) URL="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    l) LIMIT="$OPTARG" ;;
    o) OFFSET="$OPTARG" ;;
    q) QUERY="$OPTARG" ;;
    v) VERBOSE=1 ;;
    j) FORMAT="json" ;;
    h) usage; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; usage >&2; exit 1 ;;
    :) echo "option -$OPTARG requires an argument" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${RAG_ASR_SKIP_CONDA:-0}" != "1" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  if [[ -n "${CONDA_SH:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV"
  fi
fi

cmd=(
  python "$ROOT/scripts/triton_hotword_client.py"
  --url "$URL"
  --model "$MODEL"
  status
  --limit "$LIMIT"
  --offset "$OFFSET"
  --format "$FORMAT"
)

if [[ -n "$QUERY" ]]; then
  cmd+=(--query "$QUERY")
fi

if [[ "$VERBOSE" -eq 1 ]]; then
  cmd+=(--verbose)
fi

exec "${cmd[@]}"
