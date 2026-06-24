"""RAG-ASR: dual-tower audio-text retrieval for ASR hotword biasing."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AmphionAudioTower": ("rag_asr.dual_tower", "AmphionAudioTower"),
    "AmphionTextTower": ("rag_asr.dual_tower", "AmphionTextTower"),
    "HotwordsRetrievalDataset": ("rag_asr.dataset", "HotwordsRetrievalDataset"),
    "build_rare_word_vocab": ("rag_asr.dataset", "build_rare_word_vocab"),
    "build_towers_from_base": ("rag_asr.dual_tower", "build_towers_from_base"),
    "infer_retrieval_lang": ("rag_asr.dataset", "infer_retrieval_lang"),
    "load_adapter_checkpoint": ("rag_asr.model_loader", "load_adapter_checkpoint"),
    "load_base_model": ("rag_asr.model_loader", "load_base_model"),
    "load_tokenizer": ("rag_asr.model_loader", "load_tokenizer"),
    "load_towers": ("rag_asr.model_loader", "load_towers"),
    "per_positive_infonce_loss": ("rag_asr.dual_tower", "per_positive_infonce_loss"),
    "retrieve_neural": ("rag_asr.infer", "retrieve_neural"),
    "retrieval_collate_fn_factory": ("rag_asr.dataset", "retrieval_collate_fn_factory"),
    "symmetric_infonce_loss": ("rag_asr.dual_tower", "symmetric_infonce_loss"),
    "tokenise_words": ("rag_asr.dataset", "tokenise_words"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    *_EXPORTS,
]

__version__ = "0.1.0"
