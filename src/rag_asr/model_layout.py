"""Filesystem layout helpers for deployable RAG-ASR model directories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_HOTWORD_ADAPTER_SUBDIR = "hotword_adapter"
DEFAULT_HOTWORD_ADAPTER_FILENAME = "best_adapter.pt"


def resolve_hotword_adapter(
    base_model_path: str | Path,
    adapter_ckpt: Optional[str | Path] = None,
    *,
    adapter_subdir: str = DEFAULT_HOTWORD_ADAPTER_SUBDIR,
    adapter_filename: str = DEFAULT_HOTWORD_ADAPTER_FILENAME,
    must_exist: bool = True,
) -> Path:
    """Resolve the hotword retrieval adapter checkpoint for a model directory.

    Deployment should prefer a self-contained layout:

    ``base_model_path / hotword_adapter / best_adapter.pt``

    An explicit ``adapter_ckpt`` remains supported for experiments and
    backwards-compatible scripts.
    """

    if adapter_ckpt:
        path = Path(adapter_ckpt).expanduser()
    else:
        path = Path(base_model_path).expanduser() / adapter_subdir / adapter_filename

    if must_exist and not path.is_file():
        raise FileNotFoundError(
            "hotword adapter checkpoint not found: "
            f"{path}. Put the adapter under "
            f"{Path(base_model_path) / adapter_subdir / adapter_filename} "
            "or set adapter_ckpt explicitly."
        )
    return path
