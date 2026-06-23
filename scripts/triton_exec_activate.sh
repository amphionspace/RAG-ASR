#!/bin/bash
# Minimal activate script required by Triton Python backend for directory execution envs.
_CONDA_PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONDA_PREFIX="$_CONDA_PREFIX"
export PATH="$_CONDA_PREFIX/bin:${PATH:-}"
