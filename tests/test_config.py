from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_asr.cli_triton_config import render_model_repo
from rag_asr.config import PROJECT_ROOT, load_config
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
            "hotword_pool_dir": str(tmp_path / "pools"),
            "seed_pool_file": str(tmp_path / "seed.txt"),
            "default_user": "default",
            "default_top_k": 50,
            "retrieval_mode": "frame_max",
            "cache_dir": str(tmp_path / "cache"),
            "batch_text": 128,
            "enrollment_store_dir": str(tmp_path / "enrollments"),
            "enrollment_cache_max_entries": 12,
            "enrollment_cache_device": "cpu",
            "enrollment_metadata_touch_interval_sec": 42,
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
    assert cfg.to_serve_kwargs()["hotword_pool_dir"] == str(tmp_path / "pools")
    assert cfg.to_triton_parameters()["default_user"] == "default"
    assert cfg.to_triton_parameters()["adapter_subdir"] == "hotword_adapter/frame_max"
    assert cfg.to_triton_parameters()["enrollment_cache_max_entries"] == "12"
    assert cfg.to_triton_parameters()["enrollment_cache_device"] == "cpu"
    assert cfg.to_triton_parameters()["enrollment_metadata_touch_interval_sec"] == "42"
    assert cfg.retrieval.retrieval_mode == "frame_max"
    assert cfg.to_serve_kwargs()["retrieval_mode"] == "frame_max"
    assert cfg.to_triton_parameters()["retrieval_mode"] == "frame_max"
    assert cfg.to_serve_kwargs()["adapter_subdir"] == "hotword_adapter/frame_max"


def test_enrollment_persistence_defaults_to_local_store(tmp_path: Path):
    data = _minimal_config(tmp_path)
    data["retrieval"].pop("enrollment_store_dir")
    data["retrieval"].pop("enrollment_cache_max_entries")
    data["retrieval"].pop("enrollment_cache_device")
    data["retrieval"].pop("enrollment_metadata_touch_interval_sec")

    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))
    expected = str(PROJECT_ROOT / "var" / "enrollments")

    assert cfg.retrieval.enrollment_store_dir == "var/enrollments"
    assert cfg.to_serve_kwargs()["enrollment_store_dir"] == expected
    assert cfg.to_triton_parameters()["enrollment_store_dir"] == expected
    assert cfg.to_serve_kwargs()["enrollment_cache_max_entries"] == 256
    assert cfg.to_serve_kwargs()["enrollment_cache_device"] == "auto"
    assert cfg.to_serve_kwargs()["enrollment_metadata_touch_interval_sec"] == 300.0


def test_compose_config_uses_container_paths():
    cfg = load_config(PROJECT_ROOT / "configs" / "serve.compose.yaml")

    assert cfg.to_serve_kwargs()["base_model_path"] == (
        "/models/rag-asr/base/amphion_1.7b_merged"
    )
    assert cfg.to_serve_kwargs()["hotword_pool_dir"] == "/data/rag-asr/hotwords"
    assert cfg.to_serve_kwargs()["enrollment_store_dir"] == "/data/rag-asr/enrollments"
    assert cfg.to_triton_parameters()["EXECUTION_ENV_PATH"] == (
        "/opt/rag-asr/triton-exec-env.tar.gz"
    )
    assert cfg.triton.backend_dir == "/opt/tritonserver/backends"


def test_runtime_constraints_do_not_pin_extras():
    constraints = PROJECT_ROOT / "docker" / "constraints-runtime.txt"

    for line in constraints.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package = stripped.split("==", 1)[0].split("<", 1)[0].split(">", 1)[0]
        assert "[" not in package
        assert "]" not in package


def test_unknown_config_key_fails(tmp_path: Path):
    data = _minimal_config(tmp_path)
    data["model"]["unexpected"] = "bad"

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write_yaml(tmp_path / "serve.yaml", data))


def test_adapter_ckpts_selects_checkpoint_from_retrieval_mode(tmp_path: Path):
    data = _minimal_config(tmp_path)
    pooled = tmp_path / "pooled.pt"
    frame_max = tmp_path / "frame_max.pt"
    data["model"]["adapter_ckpts"] = {
        "pooled": str(pooled),
        "frame_max": str(frame_max),
    }

    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))

    assert cfg.to_serve_kwargs()["adapter_ckpt"] == str(frame_max)
    assert cfg.to_triton_parameters()["adapter_ckpt"] == str(frame_max)

    data["retrieval"]["retrieval_mode"] = "pooled"
    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))

    assert cfg.to_serve_kwargs()["adapter_ckpt"] == str(pooled)
    assert cfg.to_triton_parameters()["adapter_ckpt"] == str(pooled)


def test_retrieval_mode_selects_adapter_subdir_without_explicit_ckpt(tmp_path: Path):
    data = _minimal_config(tmp_path)

    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))

    assert cfg.to_serve_kwargs()["adapter_ckpt"] is None
    assert cfg.to_serve_kwargs()["adapter_subdir"] == "hotword_adapter/frame_max"
    assert cfg.to_triton_parameters()["adapter_subdir"] == "hotword_adapter/frame_max"

    data["retrieval"]["retrieval_mode"] = "pooled"
    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))

    assert cfg.to_serve_kwargs()["adapter_subdir"] == "hotword_adapter/pooled"
    assert cfg.to_triton_parameters()["adapter_subdir"] == "hotword_adapter/pooled"


def test_legacy_retrieval_mode_aliases_still_select_adapters(tmp_path: Path):
    data = _minimal_config(tmp_path)
    pooled = tmp_path / "pooled.pt"
    frame_max = tmp_path / "frame_max.pt"
    data["retrieval"]["retrieval_mode"] = "local"
    data["model"]["adapter_ckpts"] = {
        "global": str(pooled),
        "local": str(frame_max),
    }

    cfg = load_config(_write_yaml(tmp_path / "serve.yaml", data))

    assert cfg.retrieval.retrieval_mode == "frame_max"
    assert cfg.to_serve_kwargs()["adapter_ckpt"] == str(frame_max)


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
    assert 'key: "hotword_pool_dir"' in config_text
    assert 'name: "HOTWORD_POOL_ID"' in config_text
    assert 'name: "CONTROL_JSON"' in config_text
    assert 'name: "ENROLLMENT_ID"' in config_text
    assert 'name: "ENROLLMENT_SCOPE_ID"' in config_text
    assert 'key: "enrollment_store_dir"' in config_text
    assert 'key: "enrollment_metadata_touch_interval_sec"' in config_text
    assert 'key: "retrieval_mode"' in config_text
    assert 'key: "adapter_ckpt"' not in config_text
    assert (rendered / "rag_asr_retrieve" / "1" / "model.py").is_file()
