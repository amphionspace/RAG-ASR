from __future__ import annotations

import threading

import torch

from rag_asr.cache import load_text_emb_cache, save_text_emb_cache
from rag_asr.hotwords import normalize_hotwords
from rag_asr.serve import RAGASRRetriever, ServeConfig


def test_normalize_hotwords_dedupes_and_filters() -> None:
    batch = normalize_hotwords([" Foo ", "foo", "上海", "上", "hello   world", 123])

    assert batch.words == ["Foo", "hello world", "上海"]
    assert batch.duplicates == ["foo"]
    assert "上" in batch.invalid
    assert "123" in batch.invalid


def test_text_embedding_cache_overwrite(tmp_path) -> None:
    path = tmp_path / "text_embs.npz"
    save_text_emb_cache(path, ["a"], torch.ones(1, 2))
    save_text_emb_cache(path, ["b"], torch.zeros(1, 2))

    words, embs = load_text_emb_cache(path)
    assert words == ["a"]
    assert torch.equal(embs, torch.ones(1, 2))

    save_text_emb_cache(path, ["b"], torch.zeros(1, 2), overwrite=True)
    words, embs = load_text_emb_cache(path)
    assert words == ["b"]
    assert torch.equal(embs, torch.zeros(1, 2))


def _fake_retriever(tmp_path) -> RAGASRRetriever:
    retriever = RAGASRRetriever.__new__(RAGASRRetriever)
    retriever.cfg = ServeConfig(
        base_model_path="base",
        adapter_ckpt="adapter",
        hotword_pool_file=str(tmp_path / "pool.txt"),
        embed_dim=4,
        cache_dir=None,
        device="cpu",
    )
    retriever._device = torch.device("cpu")
    retriever._pool_lock = threading.RLock()
    retriever.hotword_pool = ["Foo", "北京"]
    retriever._pool_embs_gpu = torch.eye(2, 4)
    retriever._cache_path = lambda: None
    retriever._encode_hotwords = lambda words: torch.ones(len(words), 4)
    return retriever


def test_retriever_add_delete_and_persist(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)

    add = retriever.add_hotwords([" foo ", "上海", "上海"])
    assert add["added"] == 1
    assert add["skipped_duplicates"] == 2
    assert retriever.hotword_pool == ["Foo", "上海", "北京"]
    assert retriever._pool_embs_gpu.shape == (3, 4)

    listing = retriever.list_hotwords(query="上", limit=10)
    assert listing["hotwords"] == ["上海"]

    delete = retriever.delete_hotwords(["FOO", "不存在"])
    assert delete["deleted"] == 1
    assert delete["missing"] == ["不存在"]
    assert retriever.hotword_pool == ["上海", "北京"]

    assert (tmp_path / "pool.txt").read_text(encoding="utf-8") == "上海\n北京\n"

