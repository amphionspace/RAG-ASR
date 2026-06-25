from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_asr.cli_triton_config import render_model_repo
from rag_asr.config import load_config
from rag_asr.model_layout import resolve_hotword_adapter


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _minimal_config(tmp_path: Path) -> dict:
    return {
        "model": {
            "base_model_path": str(tmp_path / "base_model"),
            "adapter_subdir": "hotword_adapter",
            "adapter_filename": "best_adapter.pt",
            "embed_dim": 512,
            "adapter_hidden_dim": 512,
        },
        "retrieval": {
            "hotword_pool_file": str(tmp_path / "pool.txt"),
            "default_top_k": 50,
            "cache_dir": str(tmp_path / "cache"),
            "batch_text": 128,
        },
        "runtime": {
            "device": "cpu",
            "cuda_visible_devices": "0",
            "num_mel_bins": 128,
        },
        "triton": {
            "model_repo": str(tmp_path / "triton_src"),
            "rendered_model_repo": str(tmp_path / "triton_rendered"),
            "exec_env": str(tmp_path / "triton-exec-env.tar.gz"),
            "http_port": 18000,
            "grpc_port": 18001,
            "metrics_port": 18002,
        },
    }


def test_load_config_interpolates_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POOL_PATH", str(tmp_path / "env_pool.txt"))
    monkeypatch.setenv("EMPTY_PATH", "")
    monkeypatch.delenv("PYENV_LINK", raising=False)
    data = _minimal_config(tmp_path)
    data["retrieval"]["hotword_pool_file"] = "${POOL_PATH}"
    data["triton"]["backend_dir"] = "${EMPTY_PATH:-/default/backends}"
    data["triton"]["python_stub_link"] = "${PYENV_LINK:-/default/pyenv}"
    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))

    assert cfg.retrieval.hotword_pool_file == str(tmp_path / "env_pool.txt")
    assert cfg.triton.backend_dir == "/default/backends"
    assert cfg.triton.python_stub_link == "/default/pyenv"
    assert cfg.triton.metrics_port == 18002
    assert cfg.to_serve_kwargs()["hotword_pool_file"] == str(tmp_path / "env_pool.txt")
    assert cfg.to_triton_parameters()["adapter_subdir"] == "hotword_adapter"


def test_unknown_config_key_fails(tmp_path: Path):
    data = _minimal_config(tmp_path)
    data["model"]["unexpected"] = "bad"

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write_yaml(tmp_path / "serve.yaml", data))


def test_resolve_hotword_adapter_prefers_explicit_path(tmp_path: Path):
    base = tmp_path / "base"
    embedded = base / "hotword_adapter" / "best_adapter.pt"
    explicit = tmp_path / "adapter.pt"
    embedded.parent.mkdir(parents=True)
    embedded.write_bytes(b"embedded")
    explicit.write_bytes(b"explicit")

    assert resolve_hotword_adapter(base) == embedded
    assert resolve_hotword_adapter(base, explicit) == explicit


def test_resolve_hotword_adapter_missing_has_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="hotword_adapter"):
        resolve_hotword_adapter(tmp_path / "missing_base")


def test_render_model_repo_uses_yaml_parameters(tmp_path: Path):
    src = tmp_path / "triton_src"
    (src / "rag_asr_retrieve" / "1").mkdir(parents=True)
    (src / "rag_asr_retrieve" / "config.pbtxt").write_text("old", encoding="utf-8")
    (src / "rag_asr_retrieve" / "1" / "model.py").write_text("# model", encoding="utf-8")

    data = _minimal_config(tmp_path)
    config_path = _write_yaml(tmp_path / "serve.yaml", data)
    rendered = render_model_repo(config_path, None)

    config_text = (rendered / "rag_asr_retrieve" / "config.pbtxt").read_text(
        encoding="utf-8"
    )
    assert 'key: "base_model_path"' in config_text
    assert str(tmp_path / "base_model") in config_text
    assert 'key: "adapter_subdir"' in config_text
    assert "hotword_adapter" in config_text
    assert 'key: "adapter_ckpt"' not in config_text
    assert (rendered / "rag_asr_retrieve" / "1" / "model.py").is_file()
