"""Backend model registration and loading."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
_REGISTERED = False


def register_backends() -> None:
    """Register vendored Amphion / Qwen3 configs with HuggingFace Auto classes."""
    global _REGISTERED
    if _REGISTERED:
        return

    from rag_asr.backends.amphion.configuration_amphion_asr import AmphionASRConfig
    from rag_asr.backends.amphion.modeling_amphion_asr import (
        AmphionASRForConditionalGeneration,
    )
    from rag_asr.backends.qwen3.configuration_qwen3_asr import Qwen3ASRConfig
    from rag_asr.backends.qwen3.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    AutoConfig.register("amphion_asr", AmphionASRConfig)
    AutoModelForCausalLM.register(AmphionASRConfig, AmphionASRForConditionalGeneration)
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModelForCausalLM.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    _REGISTERED = True


def detect_model_type(base_model_path: str | Path) -> str:
    with open(Path(base_model_path) / "config.json", encoding="utf-8") as f:
        return json.load(f).get("model_type", "")


def load_base_model(
    base_model_path: str | Path,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float16,
    serialize_load: bool = False,
) -> nn.Module:
    """Load a frozen AmphionASR or Qwen3ASR checkpoint from vendored backends."""
    register_backends()
    base_model_path = str(base_model_path)
    model_type = detect_model_type(base_model_path).lower()

    def _load():
        if "qwen3" in model_type:
            from rag_asr.backends.qwen3.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )

            return Qwen3ASRForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
            ).to(device)
        from rag_asr.backends.amphion.configuration_amphion_asr import AmphionASRConfig
        from rag_asr.backends.amphion.modeling_amphion_asr import (
            AmphionASRForConditionalGeneration,
        )

        cfg = AmphionASRConfig.from_pretrained(base_model_path)
        return AmphionASRForConditionalGeneration.from_pretrained(
            base_model_path,
            config=cfg,
            dtype=dtype,
        ).to(device)

    if serialize_load:
        lock_path = Path(base_model_path) / ".from_pretrained.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            logger.info("loading base model (serialized): %s", base_model_path)
            model = _load()
            fcntl.flock(lockf, fcntl.LOCK_UN)
    else:
        model = _load()

    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_tokenizer(base_model_path: str | Path):
    return AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)


def load_towers(
    base_model_path: str | Path,
    adapter_ckpt: Optional[str | Path],
    *,
    embed_dim: int = 512,
    adapter_hidden_dim: Optional[int] = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    serialize_load: bool = False,
) -> Tuple[nn.Module, nn.Module, nn.Module, AutoTokenizer]:
    from rag_asr.dual_tower import AmphionAudioTower, AmphionTextTower

    if isinstance(device, str):
        device = torch.device(device)

    base_model = load_base_model(
        base_model_path, device, dtype=dtype, serialize_load=serialize_load,
    )
    tokenizer = load_tokenizer(base_model_path)

    audio_tower = AmphionAudioTower(
        base_model,
        embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
    ).to(device)
    text_tower = AmphionTextTower(
        base_model,
        embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
    ).to(device)

    if adapter_ckpt:
        load_adapter_checkpoint(audio_tower, text_tower, adapter_ckpt)

    audio_tower.eval()
    text_tower.eval()
    return base_model, audio_tower, text_tower, tokenizer


def load_adapter_checkpoint(
    audio_tower: nn.Module,
    text_tower: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    ckpt = torch.load(path, map_location=map_location)
    audio_tower.adapter.load_state_dict(ckpt["audio_adapter"])
    text_tower.adapter.load_state_dict(ckpt["text_adapter"])
    if "audio_pool" in ckpt and getattr(audio_tower, "pool", None) is not None:
        audio_tower.pool.load_state_dict(ckpt["audio_pool"])
    if "text_pool" in ckpt and getattr(text_tower, "pool", None) is not None:
        text_tower.pool.load_state_dict(ckpt["text_pool"])
