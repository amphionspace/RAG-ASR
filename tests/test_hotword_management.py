from __future__ import annotations

import json
import threading
from collections import OrderedDict

import numpy as np
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
        enrollment_store_dir=str(tmp_path / "enrollments"),
        enrollment_cache_max_entries=4,
        enrollment_cache_device="cpu",
    )
    retriever._device = torch.device("cpu")
    retriever.default_user = "default"
    retriever._pools_lock = threading.RLock()
    retriever._pools = {}
    retriever._enrollment_lock = threading.RLock()
    retriever._enrollment_cache = OrderedDict()
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


def test_enrollment_upsert_get_delete_persists_tensor(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)
    retriever._project_audio = lambda wav, sample_rate: np.ones(
        (3, 4),
        dtype=np.float32,
    )

    summary = retriever.upsert_enrollment(
        np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        user_id="tenant-a",
        enrollment_id="speaker-1",
    )

    tensor_path = tmp_path / "enrollments" / "tenant-a" / "speaker-1.pt"
    meta_path = tmp_path / "enrollments" / "tenant-a" / "speaker-1.json"
    assert summary["status"] == "ok"
    assert tensor_path.is_file()
    assert meta_path.is_file()
    assert torch.equal(torch.load(tensor_path), torch.ones(3, 4))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["user_id"] == "tenant-a"
    assert meta["enrollment_id"] == "speaker-1"
    assert meta["shape"] == [3, 4]

    retriever._enrollment_cache.clear()
    fetched = retriever.get_enrollment(user_id="tenant-a", enrollment_id="speaker-1")
    assert fetched["status"] == "ok"
    assert fetched["shape"] == [3, 4]
    assert fetched["cache_hit"] is False

    deleted = retriever.delete_enrollment(user_id="tenant-a", enrollment_id="speaker-1")
    assert deleted["deleted"] is True
    assert not tensor_path.exists()
    assert not meta_path.exists()


def test_enrollment_rejects_unsafe_id(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)

    with pytest.raises(ValueError, match="ENROLLMENT_ID"):
        retriever.get_enrollment(user_id="default", enrollment_id="../escape")


def test_results_can_disable_hotword_recall(tmp_path) -> None:
    retriever = _fake_retriever(tmp_path)
    features = torch.zeros(1, 2, 4)
    feat_lens = torch.tensor([2], dtype=torch.long)

    class _AudioTower:
        def forward_with_projector(self, features, feat_lens):
            pooled = torch.ones(1, 4)
            proj = torch.ones(1, 2, 4)
            proj_lens = torch.tensor([2], dtype=torch.long)
            return pooled, proj, proj_lens

    retriever.audio_tower = _AudioTower()

    result = retriever._results_from_features(
        features,
        feat_lens,
        top_k_values=[2],
        user_ids=["default"],
        enable_hotword_recalls=[False],
    )[0]

    assert result.word_list == []
    assert result.projector_len == 2
