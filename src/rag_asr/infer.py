"""Neural dual-tower hotword retrieval (audio → text similarity)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from rag_asr.cache import (
    load_biasing_tsv,
    load_hotword_pool_file,
    load_text_emb_cache,
    save_text_emb_cache,
    text_emb_cache_path,
)
from rag_asr.dataset import tokenise_words
from rag_asr.model_loader import load_towers

logger = logging.getLogger(__name__)


def compute_hotword_recall(real: list[str], retrieved: list[str]) -> Optional[float]:
    """Per-utterance recall@K: |real ∩ retrieved| / |real|.

    Matches Amphion ``compute_wer.compute_hotword_metrics``: strip whitespace,
    case-sensitive string intersection.  Returns ``None`` when ``real`` is empty.
    """
    real_set = {h.strip() for h in real if isinstance(h, str) and h.strip()}
    if not real_set:
        return None
    retrieved_set = {w.strip() for w in retrieved if isinstance(w, str) and w.strip()}
    return len(real_set & retrieved_set) / len(real_set)


def _hotword_hit_sets(real: list[str], retrieved: list[str]) -> tuple[set[str], set[str]]:
    """Normalise hotword lists the same way Amphion recall@K does."""
    real_set = {h.strip() for h in real if isinstance(h, str) and h.strip()}
    retrieved_set = {w.strip() for w in retrieved if isinstance(w, str) and w.strip()}
    return real_set, retrieved_set


def aggregate_recall_stats(hw_map: dict[str, dict]) -> dict[str, float | int]:
    """Micro Recall@K + PrRR, aligned with Amphion ``compute_hotword_metrics``."""
    n_total = len(hw_map)
    n_with_real = 0
    n_no_real = 0
    recall_hit = 0
    recall_total_gt = 0
    n_retrieve_perfect = 0
    n_zero_hit = 0

    for rec in hw_map.values():
        real = list(rec.get("real") or [])
        retrieved = list(rec.get("retrieved") or [])
        real_set, retrieved_set = _hotword_hit_sets(real, retrieved)
        if not real_set:
            n_no_real += 1
            continue

        n_with_real += 1
        hit = len(real_set & retrieved_set)
        recall_hit += hit
        recall_total_gt += len(real_set)
        if real_set <= retrieved_set:
            n_retrieve_perfect += 1
        if hit == 0:
            n_zero_hit += 1

    recall_at_k = (
        100.0 * recall_hit / recall_total_gt if recall_total_gt > 0 else 0.0
    )
    prrr = (
        100.0 * n_retrieve_perfect / n_with_real if n_with_real > 0 else 0.0
    )
    return {
        "n_total": n_total,
        "n_with_real": n_with_real,
        "n_no_real": n_no_real,
        "recall_at_k": recall_at_k,
        "recall_hit": recall_hit,
        "recall_total_gt": recall_total_gt,
        "n_retrieve_perfect": n_retrieve_perfect,
        "prrr": prrr,
        "n_zero_hit": n_zero_hit,
    }


def recall_summary_txt(hw_map: dict[str, dict], *, top_k: Optional[int] = None) -> str:
    """Format overall recall statistics as plain text."""
    stats = aggregate_recall_stats(hw_map)
    lines = ["# hotword retrieval recall summary", ""]
    if top_k is not None:
        lines.append(f"top_k: {top_k}")
    lines.extend([
        f"utterances_total: {stats['n_total']}",
        f"utterances_with_real: {stats['n_with_real']}",
        f"utterances_no_real: {stats['n_no_real']}",
        f"recall_at_k: {stats['recall_at_k']:.6f}",
        f"recall_hit: {stats['recall_hit']}",
        f"recall_total_gt: {stats['recall_total_gt']}",
        f"n_retrieve_perfect: {stats['n_retrieve_perfect']}",
        f"prrr: {stats['prrr']:.6f}",
        f"zero_hit (recall=0.0): {stats['n_zero_hit']}",
        "",
        "# recall_at_k = 100 * recall_hit / recall_total_gt  (micro, matches Amphion xlsx)",
        "# prrr = 100 * n_retrieve_perfect / utterances_with_real",
        "# Matching: strip whitespace, case-sensitive string intersection.",
    ])
    return "\n".join(lines) + "\n"


def write_hw_map_jsonl(path: str | Path, hw_map: dict[str, dict]) -> None:
    """Write retrieval results as JSONL (one utterance per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in hw_map:
            json.dump(hw_map[uid], f, ensure_ascii=False)
            f.write("\n")


def load_hw_map_jsonl(path: str | Path) -> dict[str, dict]:
    """Load JSONL retrieval results into ``{utt_id: record}``."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("id")
            if not uid:
                raise ValueError(f"{path}:{lineno}: missing 'id' field")
            out[uid] = rec
    return out


def write_recall_summary_txt(
    path: str | Path,
    hw_map: dict[str, dict],
    *,
    top_k: Optional[int] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recall_summary_txt(hw_map, top_k=top_k), encoding="utf-8")


def retrieve_neural(
    items: list[dict],
    hotword_pool: list[str],
    top_k_max: int,
    *,
    adapter_ckpt: str,
    base_model_path: str,
    embed_dim: int = 512,
    adapter_hidden_dim: Optional[int] = None,
    hotword_pool_file: Optional[str] = None,
    biasing_tsv_file: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 16,
    batch_text: int = 512,
    num_mel_bins: int = 128,
    cache_dir: Optional[str | Path] = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, dict]:
    """Retrieve top-K hotwords per utterance using trained dual-tower adapters.

  Each ``item`` must contain ``id``, ``mixed_audio``, optional ``start`` /
  ``duration``, and ``hotwords`` (ground truth, preserved in output).

  Returns ``{utt_id: {id, retrieved, all, real, distractor, recall}}``.
    """
    if top_k_max <= 0:
        raise ValueError(f"retrieve_neural: top_k_max must be > 0, got {top_k_max!r}.")

    per_item_candidates: Optional[dict[str, list[str]]] = None
    pool_word_to_idx: Optional[dict[str, int]] = None

    if biasing_tsv_file:
        per_item_candidates = load_biasing_tsv(biasing_tsv_file)
        hotword_pool = sorted({
            w for cands in per_item_candidates.values() for w in cands
        })
        logger.info(
            "per-item biasing TSV: %d unique words across %d utts",
            len(hotword_pool), len(per_item_candidates),
        )
    elif hotword_pool_file:
        hotword_pool = load_hotword_pool_file(hotword_pool_file)
        logger.info("hotword pool from file: %s (%d words)", hotword_pool_file, len(hotword_pool))
    else:
        logger.info("hotword pool from items: %d words", len(hotword_pool))

    cache_path = text_emb_cache_path(
        cache_dir,
        adapter_ckpt=adapter_ckpt,
        biasing_tsv_file=biasing_tsv_file,
        hotword_pool_file=hotword_pool_file,
    )

    pool_embs = None
    if cache_path is not None and cache_path.exists():
        cached_words, pool_embs = load_text_emb_cache(cache_path)
        if cached_words != hotword_pool:
            logger.warning(
                "text emb cache word mismatch (%d vs %d) — re-encoding",
                len(cached_words), len(hotword_pool),
            )
            pool_embs = None

    _device = torch.device(device)
    _, audio_tower, text_tower, tokenizer = load_towers(
        base_model_path,
        adapter_ckpt,
        embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
        device=_device,
        serialize_load=(num_shards > 1),
    )

    tag = f"shard {shard_id}/{num_shards} " if num_shards > 1 else ""
    logger.info(
        "%stowers ready. pool=%d  top_k_max=%d  items=%d",
        tag, len(hotword_pool), top_k_max, len(items),
    )

    from lhotse.features import WhisperFbank, WhisperFbankConfig

    fbank = WhisperFbank(WhisperFbankConfig(num_filters=num_mel_bins))

    if pool_embs is None:
        from rag_asr.cache import text_emb_cache_lock

        def _encode_pool() -> torch.Tensor:
            logger.info("encoding hotword pool (%d words)…", len(hotword_pool))
            parts: list[torch.Tensor] = []
            with torch.no_grad():
                for i in range(0, len(hotword_pool), batch_text):
                    chunk = hotword_pool[i : i + batch_text]
                    ids, mask = tokenise_words(chunk, tokenizer, device=_device)
                    parts.append(text_tower(ids, mask).cpu())
            embs = torch.cat(parts, dim=0)
            if cache_path is not None:
                save_text_emb_cache(cache_path, hotword_pool, embs, acquire_lock=False)
            return embs

        if cache_path is not None:
            with text_emb_cache_lock(cache_path):
                if cache_path.exists():
                    cached_words, pool_embs = load_text_emb_cache(cache_path)
                    if cached_words != hotword_pool:
                        pool_embs = _encode_pool()
                else:
                    pool_embs = _encode_pool()
        else:
            pool_embs = _encode_pool()

    pool_embs_gpu = pool_embs.to(_device, non_blocking=True)
    topk_k = min(top_k_max, len(hotword_pool))

    if per_item_candidates is not None:
        pool_word_to_idx = {w: i for i, w in enumerate(hotword_pool)}

    _faiss_index = None
    if per_item_candidates is None:
        try:
            import faiss  # type: ignore

            _faiss_index = faiss.IndexFlatIP(pool_embs.shape[1])
            _faiss_index.add(pool_embs.numpy().astype("float32"))
            logger.info("FAISS IndexFlatIP built (%d words)", len(hotword_pool))
        except ImportError:
            logger.info("faiss not installed, using batched torch matmul on GPU")

    hw_map: dict[str, dict] = {}
    n_total = len(items)
    t0 = time.time()
    prog_tag = f"s{shard_id} " if num_shards > 1 else ""

    for batch_start in range(0, n_total, batch_size):
        batch_items = items[batch_start : batch_start + batch_size]
        feats_list: list[torch.Tensor] = []
        valid_items: list[dict] = []

        for it in batch_items:
            try:
                audio_path = it["mixed_audio"]
                start = float(it.get("start") or 0.0)
                duration = it.get("duration")

                import soundfile as sf

                offset_samples = None
                num_samples = None
                if duration is not None:
                    with sf.SoundFile(audio_path) as f:
                        sr = f.samplerate
                        offset_samples = int(start * sr)
                        num_samples = int(float(duration) * sr)
                        f.seek(offset_samples)
                        audio = f.read(num_samples, dtype="float32", always_2d=False)
                else:
                    audio, sr = sf.read(
                        audio_path,
                        start=int(start * sf.info(audio_path).samplerate)
                        if start else 0,
                        dtype="float32",
                        always_2d=False,
                    )
                if sr != 16000:
                    import torchaudio

                    audio_t = torch.from_numpy(audio).unsqueeze(0)
                    audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
                    audio = audio_t.squeeze(0).numpy()
                    sr = 16000
                feat = fbank.extract(samples=audio, sampling_rate=sr)
                feats_list.append(torch.from_numpy(feat))
                valid_items.append(it)
            except Exception as e:
                logger.warning("skipping %s: %s", it.get("id"), e)

        if not feats_list:
            continue

        feat_lens = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
        max_t = int(feat_lens.max().item())
        n_feats = feats_list[0].shape[1]
        features = torch.zeros(len(feats_list), max_t, n_feats, dtype=torch.float32)
        for i, f in enumerate(feats_list):
            features[i, : f.shape[0], :] = f

        features = features.to(_device)
        feat_lens = feat_lens.to(_device)

        with torch.no_grad():
            a_embs = audio_tower(features, feat_lens)

        batch_indices: list[list[int]] | None = None
        if per_item_candidates is None and _faiss_index is not None:
            import faiss  # type: ignore

            _, batch_idx = _faiss_index.search(
                a_embs.cpu().numpy().astype("float32"), topk_k,
            )
            batch_indices = [row.tolist() for row in batch_idx]
        elif per_item_candidates is None:
            with torch.no_grad():
                batch_scores = a_embs @ pool_embs_gpu.T
                batch_indices = batch_scores.topk(topk_k, dim=1).indices.cpu().tolist()

        for i, it in enumerate(valid_items):
            uid = it["id"]
            real_hw = list(it.get("hotwords") or [])
            real_set = {h.lower() for h in real_hw}

            if per_item_candidates is not None and pool_word_to_idx is not None:
                item_cands = per_item_candidates.get(uid)
                if item_cands:
                    g_indices = [
                        pool_word_to_idx[w] for w in item_cands if w in pool_word_to_idx
                    ]
                    if g_indices:
                        item_scores = (
                            a_embs[i].unsqueeze(0) @ pool_embs_gpu[g_indices].T
                        ).squeeze(0)
                        local_k = min(top_k_max, len(g_indices))
                        local_idx = item_scores.topk(local_k).indices.tolist()
                        retrieved = [item_cands[j] for j in local_idx]
                    else:
                        retrieved = []
                else:
                    scores = (a_embs[i].unsqueeze(0) @ pool_embs_gpu.T).squeeze(0)
                    local_k = min(top_k_max, scores.shape[0])
                    retrieved = [
                        hotword_pool[j] for j in scores.topk(local_k).indices.tolist()
                    ]
            elif batch_indices is not None:
                retrieved = [hotword_pool[j] for j in batch_indices[i]]
            else:
                retrieved = []

            hw_map[uid] = {
                "id": uid,
                "retrieved": retrieved,
                "all": sorted(set(retrieved) | set(real_hw)),
                "real": real_hw,
                "distractor": [w for w in retrieved if w.lower() not in real_set],
                "recall": compute_hotword_recall(real_hw, retrieved),
            }

        n_done = batch_start + len(valid_items)
        if n_done % 200 == 0 or n_done >= n_total:
            elapsed = time.time() - t0
            speed = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / speed if speed > 0 else 0
            sys.stdout.write(
                f"\r  [retrieve_neural] {prog_tag}[{n_done}/{n_total}] "
                f"{speed:.1f} it/s  ETA={eta:.0f}s"
            )
            sys.stdout.flush()

    print()
    logger.info(
        "retrieve_neural done in %.1fs (top_k_max=%d, %d items, pool=%d)",
        time.time() - t0, top_k_max, n_total, len(hotword_pool),
    )
    return hw_map
