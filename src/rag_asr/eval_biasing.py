#!/usr/bin/env python3
"""Per-utterance biasing-list recall evaluation for the dual-tower retrieval model.

For each TSV file (format: utt_id | text | hotwords_json | biasing_list_json),
the script:
  1. Encodes each utterance's audio with the audio tower.
  2. Encodes all unique words in the biasing lists with the text tower.
  3. For each utterance ranks its per-utterance biasing list by cosine similarity
     to the audio embedding and reports Recall@K.

Usage
-----
python src/retrieval/test_biasing_recall.py \
    --base-model-path  /path/to/checkpoint_merged \
    --adapter-path     exp/retrieval/xxx/best_adapter.pt \
    --embed-dim        1024 \
    --tsv-clean        ref/test-clean.biasing_100.tsv  ref/test-clean.biasing_500.tsv \
                       ref/test-clean.biasing_1000.tsv ref/test-clean.biasing_2000.tsv \
    --tsv-other        ref/test-other.biasing_100.tsv  ref/test-other.biasing_500.tsv \
                       ref/test-other.biasing_1000.tsv ref/test-other.biasing_2000.tsv \
    --recordings-clean /manifests/librispeech_recordings_test-clean.jsonl.gz \
    --recordings-other /manifests/librispeech_recordings_test-other.jsonl.gz \
    --recall-k  1 2 3 4 5 10 15 20 50 \
    --batch-audio 32 \
    --batch-text  512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Prevent HuggingFace tokenizers from triggering fork-after-parallelism warnings
# when DataLoader spawns worker processes after the tokenizer has been used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

def parse_tsv(path: str) -> List[Dict]:
    """Parse biasing TSV.  Each row → dict with keys:
      utt_id, text, hotwords (list[str]), biasing_list (list[str])
    Skips rows with empty hotwords (nothing to recall).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            utt_id, text, hw_json, bias_json = parts[0], parts[1], parts[2], parts[3]
            hotwords = json.loads(hw_json)
            biasing  = json.loads(bias_json)
            if not hotwords or not biasing:
                continue
            rows.append({
                "utt_id":       utt_id,
                "text":         text,
                "hotwords":     [w.lower() for w in hotwords],
                "biasing_list": [w.lower() for w in biasing],
            })
    return rows


# ---------------------------------------------------------------------------
# Audio dataset backed by lhotse recordings
# ---------------------------------------------------------------------------

class RecordingDataset(Dataset):
    """Load recordings by ID from a lhotse RecordingSet."""

    def __init__(self, utt_ids: List[str], recordings_map: dict, fbank):
        self.utt_ids = utt_ids
        self.recordings_map = recordings_map
        self.fbank = fbank

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        rec = self.recordings_map[utt_id]
        audio = rec.load_audio()             # (1, T_samples) float32
        audio = audio.squeeze(0)             # (T_samples,)
        feats = self.fbank.extract(audio, rec.sampling_rate)  # (T_frames, n_mels)
        return utt_id, torch.from_numpy(feats).float()


def audio_collate(batch):
    utt_ids, feats_list = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
    max_t   = lengths.max().item()
    padded  = torch.zeros(len(feats_list), max_t, feats_list[0].shape[1])
    for i, f in enumerate(feats_list):
        padded[i, :f.shape[0]] = f
    return list(utt_ids), padded, lengths


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_audio_for_split(
    utt_ids: List[str],
    audio_tower,
    recordings_map: dict,
    fbank,
    device: torch.device,
    batch_audio: int = 32,
) -> Dict[str, torch.Tensor]:
    """Encode all utterances in *utt_ids* once and return a {utt_id: emb} dict.

    Call this once per split (test-clean / test-other) and reuse the cache
    across all biasing-list sizes so that each audio is encoded only once.
    """
    ds     = RecordingDataset(utt_ids, recordings_map, fbank)
    loader = DataLoader(ds, batch_size=batch_audio, collate_fn=audio_collate,
                        num_workers=4, pin_memory=True)
    utt_embs: Dict[str, torch.Tensor] = {}
    for utt_ids_batch, feats, lengths in loader:
        feats   = feats.to(device)
        lengths = lengths.to(device)
        embs    = audio_tower(feats, lengths).cpu()  # (B, D)
        for uid, emb in zip(utt_ids_batch, embs):
            utt_embs[uid] = emb
    return utt_embs


@torch.no_grad()
def evaluate_tsv(
    tsv_path: str,
    audio_tower,
    text_tower,
    recordings_map: dict,
    fbank,
    tokenizer,
    ks: List[int],
    device: torch.device,
    batch_audio: int = 32,
    batch_text:  int = 512,
    utt_embs_cache: Dict[str, torch.Tensor] | None = None,
) -> Dict[int, float]:
    from rag_asr.dataset import tokenise_words

    rows = parse_tsv(tsv_path)
    if not rows:
        logger.warning("No valid rows in %s", tsv_path)
        return {k: 0.0 for k in ks}

    # Filter to rows whose recording exists
    missing = [r["utt_id"] for r in rows if r["utt_id"] not in recordings_map]
    if missing:
        logger.warning("%d utterances not found in recordings (e.g. %s); skipping.",
                       len(missing), missing[:3])
        rows = [r for r in rows if r["utt_id"] in recordings_map]
    if not rows:
        return {k: 0.0 for k in ks}

    # ---- 1. Pre-encode all unique words in this TSV -------------------------
    all_words_set: set[str] = set()
    for r in rows:
        all_words_set.update(r["biasing_list"])
    all_words = sorted(all_words_set)
    word2idx  = {w: i for i, w in enumerate(all_words)}

    logger.info("  Encoding %d unique words…", len(all_words))
    word_embs_parts = []
    for i in range(0, len(all_words), batch_text):
        chunk = all_words[i : i + batch_text]
        ids, mask = tokenise_words(chunk, tokenizer, device=device)
        word_embs_parts.append(text_tower(ids, mask).cpu())
    word_embs = torch.cat(word_embs_parts, dim=0)  # (W, D)

    # ---- 2. Audio embeddings (use cache if available) -----------------------
    if utt_embs_cache is not None:
        utt_embs = utt_embs_cache
    else:
        logger.info("  Encoding %d utterances…", len(rows))
        utt_ids_list = [r["utt_id"] for r in rows]
        utt_embs = encode_audio_for_split(
            utt_ids_list, audio_tower, recordings_map, fbank, device, batch_audio
        )

    # ---- 3. Compute recall@K ------------------------------------------------
    hits  = {k: 0.0 for k in ks}
    total = 0
    max_k = max(ks)

    for r in rows:
        uid  = r["utt_id"]
        if uid not in utt_embs:
            continue
        a_emb = utt_embs[uid]                    # (D,)
        b_idx  = [word2idx[w] for w in r["biasing_list"]]
        b_embs = word_embs[b_idx]                # (N_bias, D)
        sims   = (a_emb.unsqueeze(0) @ b_embs.T).squeeze(0)  # (N_bias,)

        topk_n = min(max_k, sims.shape[0])
        topk_local = sims.topk(topk_n).indices.tolist()

        gt_set = set(r["hotwords"])
        for k in ks:
            retrieved = {r["biasing_list"][topk_local[j]] for j in range(min(k, topk_n))}
            hits[k] += len(gt_set & retrieved) / len(gt_set)
        total += 1

    recall = {k: hits[k] / total if total > 0 else 0.0 for k in ks}
    logger.info("  Total utterances evaluated: %d", total)
    return recall


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate biasing-list recall for dual-tower retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-path",    required=True,
                   help="Path to best_adapter.pt / last_adapter.pt")
    p.add_argument("--embed-dim",        type=int, default=1024)
    p.add_argument("--adapter-hidden-dim", type=int, default=None,
                   help="Hidden dim of MLP adapter (None = same as embed_dim*2.5)")
    p.add_argument("--tsv-clean",       nargs="+", default=[],
                   help="TSV files for test-clean (biasing_100/500/1000/2000)")
    p.add_argument("--tsv-other",       nargs="+", default=[],
                   help="TSV files for test-other")
    p.add_argument("--recordings-clean", default=None,
                   help="Lhotse recordings manifest for test-clean")
    p.add_argument("--recordings-other", default=None,
                   help="Lhotse recordings manifest for test-other")
    p.add_argument("--recall-k",        nargs="+", type=int,
                   default=[1, 2, 3, 4, 5, 10, 15, 20, 50])
    p.add_argument("--batch-audio",     type=int, default=32)
    p.add_argument("--batch-text",      type=int, default=512)
    p.add_argument("--num-mel-bins",    type=int, default=128)
    p.add_argument("--fp16",            action="store_true")
    p.add_argument("--gpu",             type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Load base model + adapters -----------------------------------------
    logger.info("Loading base model from %s …", args.base_model_path)
    from rag_asr.dual_tower import AmphionAudioTower, AmphionTextTower
    from rag_asr.model_loader import load_base_model, load_tokenizer, load_adapter_checkpoint

    dtype = torch.float16 if args.fp16 else torch.float32
    base_model = load_base_model(args.base_model_path, device, dtype=dtype)
    tokenizer = load_tokenizer(args.base_model_path)

    audio_tower = AmphionAudioTower(base_model, embed_dim=args.embed_dim,
                                    adapter_hidden_dim=args.adapter_hidden_dim).to(device)
    text_tower  = AmphionTextTower(base_model,  embed_dim=args.embed_dim,
                                   adapter_hidden_dim=args.adapter_hidden_dim).to(device)

    logger.info("Loading adapter weights from %s …", args.adapter_path)
    load_adapter_checkpoint(audio_tower, text_tower, args.adapter_path, map_location=device)
    audio_tower.eval()
    text_tower.eval()

    # ---- Fbank extractor ----------------------------------------------------
    from lhotse.features import WhisperFbank, WhisperFbankConfig
    fbank = WhisperFbank(WhisperFbankConfig(num_filters=args.num_mel_bins))

    # ---- Build recordings maps ----------------------------------------------
    def _build_recordings_map(manifest_path: str) -> dict:
        from lhotse import RecordingSet
        logger.info("Loading recordings manifest: %s", manifest_path)
        recs = RecordingSet.from_jsonl(manifest_path)
        return {r.id: r for r in recs}

    # ---- Run evaluation -----------------------------------------------------
    ks = sorted(args.recall_k)
    all_results = []   # list of (label, recall_dict)

    for split_name, tsv_files, rec_manifest in [
        ("test-clean", args.tsv_clean, args.recordings_clean),
        ("test-other", args.tsv_other, args.recordings_other),
    ]:
        if not tsv_files:
            continue
        if not rec_manifest:
            logger.warning("No recordings manifest provided for %s, skipping.", split_name)
            continue

        recordings_map = _build_recordings_map(rec_manifest)

        # Pre-encode every unique utterance in this split ONCE,
        # then reuse the cache across all biasing-list sizes.
        all_utt_ids: list[str] = []
        seen: set[str] = set()
        for tsv_path in tsv_files:
            for r in parse_tsv(tsv_path):
                uid = r["utt_id"]
                if uid not in seen and uid in recordings_map:
                    all_utt_ids.append(uid)
                    seen.add(uid)
        logger.info("=== %s: pre-encoding %d unique utterances (shared across %d TSVs) ===",
                    split_name, len(all_utt_ids), len(tsv_files))
        utt_embs_cache = encode_audio_for_split(
            all_utt_ids, audio_tower, recordings_map, fbank, device, args.batch_audio
        )

        for tsv_path in tsv_files:
            n_label = Path(tsv_path).stem.split("biasing_")[-1]
            label   = f"{split_name}  N={n_label}"
            logger.info("=== Evaluating %s ===", label)

            recall = evaluate_tsv(
                tsv_path        = tsv_path,
                audio_tower     = audio_tower,
                text_tower      = text_tower,
                recordings_map  = recordings_map,
                fbank           = fbank,
                tokenizer       = tokenizer,
                ks              = ks,
                device          = device,
                batch_audio     = args.batch_audio,
                batch_text      = args.batch_text,
                utt_embs_cache  = utt_embs_cache,
            )
            all_results.append((label, recall))

    # ---- Print summary table ------------------------------------------------
    if not all_results:
        logger.warning("No results to display.")
        return

    k_header = "  ".join(f"R@{k:>2}" for k in ks)
    sep = "-" * (22 + len(k_header) + 4)
    print()
    print(sep)
    print(f"{'Split / N':22s}  {k_header}")
    print(sep)
    for label, recall in all_results:
        vals = "  ".join(f"{recall[k]*100:5.1f}" for k in ks)
        print(f"{label:22s}  {vals}")
    print(sep)


if __name__ == "__main__":
    main()
