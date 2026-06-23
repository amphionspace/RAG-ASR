"""RAG-ASR: dual-tower audio-text retrieval for ASR hotword biasing."""

from rag_asr.dual_tower import (  # noqa: F401
    AmphionAudioTower,
    AmphionTextTower,
    build_towers_from_base,
    per_positive_infonce_loss,
    symmetric_infonce_loss,
)
from rag_asr.dataset import (  # noqa: F401
    HotwordsRetrievalDataset,
    build_rare_word_vocab,
    infer_retrieval_lang,
    retrieval_collate_fn_factory,
    tokenise_words,
)
from rag_asr.infer import retrieve_neural  # noqa: F401
from rag_asr.model_loader import (  # noqa: F401
    load_adapter_checkpoint,
    load_base_model,
    load_towers,
    load_tokenizer,
)

__all__ = [
    "AmphionAudioTower",
    "AmphionTextTower",
    "HotwordsRetrievalDataset",
    "build_rare_word_vocab",
    "build_towers_from_base",
    "infer_retrieval_lang",
    "load_adapter_checkpoint",
    "load_base_model",
    "load_tokenizer",
    "load_towers",
    "per_positive_infonce_loss",
    "retrieve_neural",
    "retrieval_collate_fn_factory",
    "symmetric_infonce_loss",
    "tokenise_words",
]

__version__ = "0.1.0"
