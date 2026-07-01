from __future__ import annotations

import threading

import pytest
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
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    (pool_dir / "default.txt").write_text("Foo\n北京\n", encoding="utf-8")

    retriever = RAGASRRetriever.__new__(RAGASRRetriever)
    retriever.cfg = ServeConfig(
        base_model_path="base",
        adapter_ckpt="adapter",
        hotword_pool_dir=str(pool_dir),
        seed_pool_file=None,
        default_user="default",
        embed_dim=4,
        cache_dir=None,
        device="cpu",
    )
    retriever._device = torch.device("cpu")
    retriever.default_user = "default"
    retriever._pools_lock = threading.RLock()
    retriever._pools = {}
    retriever.adapter_ckpt = "adapter"
    retriever._encode_hotwords = lambda words: torch.ones(len(words), 4)
    return retriever


def test_retriever_add_delete_and_persist(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)

    add = retriever.add_hotwords([" foo ", "上海", "上海"])
    assert add["added"] == 1
    assert add["skipped_duplicates"] == 2
    default_pool = retriever._get_or_load_pool("default")
    assert default_pool.words == ["Foo", "上海", "北京"]
    assert default_pool.embs_gpu.shape == (3, 4)

    listing = retriever.list_hotwords(query="上", limit=10)
    assert listing["hotwords"] == ["上海"]

    delete = retriever.delete_hotwords(["FOO", "不存在"])
    assert delete["deleted"] == 1
    assert delete["missing"] == ["不存在"]
    assert default_pool.words == ["上海", "北京"]

    assert (tmp_path / "pools" / "default.txt").read_text(encoding="utf-8") == "上海\n北京\n"


def test_retriever_isolates_hotwords_by_user(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)

    retriever.add_hotwords(["上海"], user_id="user-a")
    retriever.add_hotwords(["深圳"], user_id="user-b")

    assert retriever.list_hotwords(user_id="user-a")["hotwords"] == ["上海"]
    assert retriever.list_hotwords(user_id="user-b")["hotwords"] == ["深圳"]
    assert retriever.list_hotwords(user_id="default")["hotwords"] == ["Foo", "北京"]

    assert (tmp_path / "pools" / "user-a.txt").read_text(encoding="utf-8") == "上海\n"
    assert (tmp_path / "pools" / "user-b.txt").read_text(encoding="utf-8") == "深圳\n"


def test_retriever_rejects_unsafe_user_id(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)

    with pytest.raises(ValueError, match="USER_ID"):
        retriever.list_hotwords(user_id="../escape")

