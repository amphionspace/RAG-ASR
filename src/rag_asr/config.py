"""Declarative configuration loader for RAG-ASR services."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, TypeVar

import yaml

from rag_asr.model_layout import (
    DEFAULT_HOTWORD_ADAPTER_FILENAME,
    DEFAULT_HOTWORD_ADAPTER_SUBDIR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "serve.yaml"
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
T = TypeVar("T")


def _interpolate_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}. Create configs/serve.yaml "
            "or set RAG_ASR_CONFIG."
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return _interpolate_env(data)


def _section(cls: type[T], data: dict[str, Any], name: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in config section {name}: {unknown}")
    return cls(**data)


def _resolve_path(value: Optional[str], *, allow_special: bool = False) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if allow_special and value.lower() in {"none", "off"}:
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


@dataclass(frozen=True)
class ModelConfig:
    base_model_path: str
    adapter_subdir: str = DEFAULT_HOTWORD_ADAPTER_SUBDIR
    adapter_filename: str = DEFAULT_HOTWORD_ADAPTER_FILENAME
    adapter_ckpt: Optional[str] = None
    embed_dim: int = 512
    adapter_hidden_dim: Optional[int] = 512


@dataclass(frozen=True)
class RetrievalConfig:
    hotword_pool_file: str
    default_top_k: int = 50
    cache_dir: Optional[str] = "var/retrieve_cache"
    batch_text: int = 512


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"
    cuda_visible_devices: str = "0"
    num_mel_bins: int = 128


@dataclass(frozen=True)
class TritonConfig:
    model_repo: str = "triton"
    rendered_model_repo: str = "var/triton_repo"
    exec_env: str = "${CONDA_PREFIX}/../triton-exec"
    backend_dir: Optional[str] = None
    python_stub_link: str = "/opt/pyenv_build/versions/3.12.3"
    http_port: int = 8000
    grpc_port: int = 8001


@dataclass(frozen=True)
class RagASRConfig:
    model: ModelConfig
    retrieval: RetrievalConfig
    runtime: RuntimeConfig = RuntimeConfig()
    triton: TritonConfig = TritonConfig()

    def to_serve_kwargs(self) -> dict[str, Any]:
        return {
            "base_model_path": _resolve_path(self.model.base_model_path),
            "hotword_pool_file": _resolve_path(self.retrieval.hotword_pool_file),
            "adapter_ckpt": _resolve_path(self.model.adapter_ckpt)
            if self.model.adapter_ckpt
            else None,
            "adapter_subdir": self.model.adapter_subdir,
            "adapter_filename": self.model.adapter_filename,
            "embed_dim": self.model.embed_dim,
            "adapter_hidden_dim": self.model.adapter_hidden_dim,
            "default_top_k": self.retrieval.default_top_k,
            "cache_dir": _resolve_path(self.retrieval.cache_dir, allow_special=True),
            "device": self.runtime.device,
            "num_mel_bins": self.runtime.num_mel_bins,
            "batch_text": self.retrieval.batch_text,
        }

    def to_triton_parameters(self) -> dict[str, str]:
        params = {
            "EXECUTION_ENV_PATH": _resolve_path(self.triton.exec_env) or "",
            "base_model_path": _resolve_path(self.model.base_model_path) or "",
            "hotword_pool_file": _resolve_path(self.retrieval.hotword_pool_file) or "",
            "adapter_subdir": self.model.adapter_subdir,
            "adapter_filename": self.model.adapter_filename,
            "embed_dim": str(self.model.embed_dim),
            "adapter_hidden_dim": ""
            if self.model.adapter_hidden_dim is None
            else str(self.model.adapter_hidden_dim),
            "default_top_k": str(self.retrieval.default_top_k),
            "cache_dir": _resolve_path(self.retrieval.cache_dir, allow_special=True) or "",
            "device": self.runtime.device,
        }
        if self.model.adapter_ckpt:
            params["adapter_ckpt"] = _resolve_path(self.model.adapter_ckpt) or ""
        return params


def load_config(path: Optional[str | Path] = None) -> RagASRConfig:
    config_path = Path(path or os.getenv("RAG_ASR_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    data = _load_yaml(config_path)
    return RagASRConfig(
        model=_section(ModelConfig, data.get("model", {}), "model"),
        retrieval=_section(RetrievalConfig, data.get("retrieval", {}), "retrieval"),
        runtime=_section(RuntimeConfig, data.get("runtime", {}), "runtime"),
        triton=_section(TritonConfig, data.get("triton", {}), "triton"),
    )
