#!/usr/bin/env python3
"""Training script for the dual-tower hotword retrieval model.

Quick start
-----------
python src/retrieval/train_retrieval.py \\
    --base-model-path /path/to/amphion_4b_checkpoint \\
    --supervisions  /path/to/manifests/librispeech_supervisions_train-other-500_hotwords.jsonl.gz \\
                    /path/to/manifests/librispeech_supervisions_train-clean-360_hotwords.jsonl.gz \\
                    /path/to/manifests/librispeech_supervisions_train-clean-100_hotwords.jsonl.gz \\
    --recordings    /path/to/manifests/librispeech_recordings_train-other-500.jsonl.gz \\
                    /path/to/manifests/librispeech_recordings_train-clean-360.jsonl.gz \\
                    /path/to/manifests/librispeech_recordings_train-clean-100.jsonl.gz \\
    --output-dir    exp/retrieval/amphion_4b_dual_tower \\
    --embed-dim 512 --batch-size 32 --num-negatives 4096 \\
    --lr 3e-4 --epochs 10

The script:
  1. Loads the Amphion-4B base model and freezes all its weights.
  2. Builds ``AmphionAudioTower`` and ``AmphionTextTower`` (only adapters
     are trainable).
  3. Builds a global rare-word vocabulary from the training manifests.
  4. Trains with symmetric InfoNCE loss (CLIP-style, in-batch negatives only):
       - One random positive hotword is selected per audio.
       - Both audio→text and text→audio directions use the same B in-batch
         candidates, keeping both directions equally difficult.
  5. Saves adapter weights to ``output-dir/best_adapter.pt`` (best
     validation loss) and ``output-dir/last_adapter.pt``.
  6. Optionally evaluates recall@K on a validation split.

Output checkpoint format
------------------------
``torch.save({"audio_adapter": audio_tower.adapter.state_dict(),
              "text_adapter":  text_tower.adapter.state_dict(),
              "embed_dim":     args.embed_dim}, path)``

To reload:
  audio_tower.adapter.load_state_dict(ckpt["audio_adapter"])
  text_tower.adapter.load_state_dict(ckpt["text_adapter"])
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Installed / editable: ``rag_asr`` is importable via PYTHONPATH=src or pip install -e .

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train dual-tower hotword retrieval model on Amphion-4B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Data ---
    p.add_argument(
        "--supervisions", nargs="+", required=True, metavar="PATH",
        help="Lhotse supervision manifest paths (hotword-annotated).",
    )
    p.add_argument(
        "--recordings", nargs="+", required=True, metavar="PATH",
        help="Lhotse recording manifest paths (matching supervisions).",
    )
    p.add_argument(
        "--val-supervisions", nargs="*", default=None, metavar="PATH",
        help="Optional validation supervision manifests.",
    )
    p.add_argument(
        "--val-recordings", nargs="*", default=None, metavar="PATH",
        help="Optional validation recording manifests.",
    )
    p.add_argument(
        "--train-vocab", default=None, metavar="PATH",
        help=(
            "Single-language negative-sampling pool (legacy).  "
            "Use --train-vocab-en / --train-vocab-zh for bilingual training."
        ),
    )
    p.add_argument(
        "--train-vocab-en", default=None, metavar="PATH",
        help="English negative-sampling pool (used for English utterances).",
    )
    p.add_argument(
        "--train-vocab-zh", default=None, metavar="PATH",
        help="Chinese negative-sampling pool (used for Chinese utterances).",
    )
    p.add_argument(
        "--val-vocab", default=None, metavar="PATH",
        help="Single retrieval pool for validation (legacy).",
    )
    p.add_argument(
        "--val-vocab-en", default=None, metavar="PATH",
        help="English retrieval pool for validation Recall@K.",
    )
    p.add_argument(
        "--val-vocab-zh", default=None, metavar="PATH",
        help="Chinese retrieval pool for validation Recall@K.",
    )
    p.add_argument(
        "--max-duration-s", type=float, default=30.0,
        help="Skip cuts longer than this (seconds).",
    )
    # --- Model ---
    p.add_argument(
        "--base-model-path", required=True, metavar="PATH",
        help="Path to the Amphion-4B HF checkpoint directory.",
    )
    p.add_argument(
        "--embed-dim", type=int, default=512,
        help="Shared embedding dimension of the dual-tower.",
    )
    p.add_argument(
        "--adapter-hidden-dim", type=int, default=None,
        help="Hidden dim of MLP adapters (default: max(in_dim, embed_dim)).",
    )
    p.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate in MLP adapters.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.07,
        help="InfoNCE temperature.",
    )
    p.add_argument(
        "--learnable-temperature", action="store_true", default=False,
        help="Make the temperature (logit scale) a learnable parameter. "
             "When set, log_scale is trained alongside the adapters and "
             "clamped to [0, log(100)] after each step.",
    )
    p.add_argument(
        "--loss-w-a2t", type=float, default=1.0,
        help="Weight for audio→text loss direction.",
    )
    p.add_argument(
        "--loss-w-t2a", type=float, default=1.0,
        help="Weight for text→audio loss direction.",
    )
    p.add_argument(
        "--resume", default=None, metavar="PATH",
        help="Path to a previously saved adapter checkpoint to resume from.",
    )
    # --- Training ---
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-negatives", type=int, default=4096,
                   help="N = total candidates per audio (1 positive + N-1 global negatives). "
                        "Each batch samples N-1 words from the training vocab (excluding the "
                        "batch's own hotwords) as shared negatives for all audios in the batch.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Number of linear warmup steps. "
                        "Overrides --warmup-ratio when both are set.")
    p.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="Fraction of total steps used for linear warmup "
                        "(e.g. 0.05 = first 5%% of training). "
                        "Ignored if --warmup-steps > 0.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default=None, metavar="IDS",
                   help="Comma-separated GPU ids to use, e.g. '0' or '0,1,2,3'. "
                        "Defaults to GPU 0 if CUDA is available, else CPU. "
                        "When multiple ids are given the adapters are wrapped in "
                        "DataParallel and the batch is split across all listed GPUs.")
    p.add_argument("--fp16", action="store_true",
                   help="Use automatic mixed precision (FP16).")
    # --- Logging / saving ---
    p.add_argument("--output-dir", required=True, metavar="PATH",
                   help="Directory for checkpoints and logs.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Log training loss every N steps.")
    p.add_argument("--val-every-epoch", type=int, default=0,
                   help="Run validation every N epochs. "
                        "Set to 0 to disable epoch-level validation "
                        "(use --val-every-steps instead).")
    p.add_argument("--val-every-steps", type=int, default=0,
                   help="Run validation every N global steps. "
                        "When both --val-every-steps and --val-every-epoch are "
                        "non-zero, both triggers are active independently. "
                        "When both are 0, validation runs once at the end of "
                        "each epoch (backward-compatible default).")
    p.add_argument("--val-recall-k", nargs="+", type=int,
                   default=[1, 2, 3, 4, 5, 10, 15, 20],
                   help="K values for Recall@K on validation set. "
                        "Best checkpoint is selected by the smallest K.")
    # --- Mel feature extractor ---
    p.add_argument("--num-mel-bins", type=int, default=128,
                   help="Number of mel filter banks (128 for Whisper-style).")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest_lazy(path: str):
    """Load a gzip or plain JSONL manifest lazily via lhotse."""
    from lhotse import load_manifest_lazy
    return load_manifest_lazy(path)


def _load_vocab_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pretokenize_vocab(
    vocab: list[str],
    tokenizer,
    max_length: int = 20,
) -> dict:
    enc = tokenizer(
        vocab,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "vocab": vocab,
        "ids_cpu": enc["input_ids"],
        "mask_cpu": enc["attention_mask"].float(),
        "index": {w: i for i, w in enumerate(vocab)},
    }


def _sample_negative_indices(
    vocab_data: dict,
    exclude_words: set[str],
    num_negatives: int,
) -> list[int]:
    """Sample up to ``num_negatives`` indices, excluding ``exclude_words``."""
    vocab_index = vocab_data["index"]
    exclude_idxs = {vocab_index[w] for w in exclude_words if w in vocab_index}
    cand = [i for i in range(len(vocab_data["vocab"])) if i not in exclude_idxs]
    n_neg = min(num_negatives, len(cand))
    if n_neg <= 0:
        return []
    return random.sample(cand, n_neg)


def _load_train_vocab_lists(args, train_cuts) -> dict[str, list[str]]:
    from rag_asr.dataset import build_rare_word_vocab

    if args.train_vocab_en and args.train_vocab_zh:
        return {
            "en": _load_vocab_file(Path(args.train_vocab_en)),
            "zh": _load_vocab_file(Path(args.train_vocab_zh)),
        }
    if args.train_vocab:
        return {"all": _load_vocab_file(Path(args.train_vocab))}
    return {"all": build_rare_word_vocab(train_cuts)}


def _load_val_vocab_lists(args, train_vocab_lists: dict[str, list[str]]) -> dict[str, list[str]]:
    if args.val_vocab_en and args.val_vocab_zh:
        return {
            "en": _load_vocab_file(Path(args.val_vocab_en)),
            "zh": _load_vocab_file(Path(args.val_vocab_zh)),
        }
    if args.val_vocab:
        return {"all": _load_vocab_file(Path(args.val_vocab))}
    if "en" in train_vocab_lists and "zh" in train_vocab_lists:
        return {"en": train_vocab_lists["en"], "zh": train_vocab_lists["zh"]}
    return {"all": train_vocab_lists["all"]}


def _load_cutset(sup_paths: list[str], rec_paths: list[str]):
    """Build a trimmed CutSet from parallel supervision + recording lists."""
    from lhotse import CutSet, load_manifest_lazy
    from lhotse.supervision import SupervisionSet
    from lhotse.audio import RecordingSet

    sups = None
    recs = None
    for s, r in zip(sup_paths, rec_paths):
        s_part = SupervisionSet.from_jsonl_lazy(s)
        r_part = RecordingSet.from_jsonl_lazy(r)
        sups = s_part if sups is None else sups + s_part
        recs = r_part if recs is None else recs + r_part

    cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
    cuts = cuts.trim_to_supervisions(keep_overlapping=False)
    # WhisperFbank expects 16 kHz; some corpora (e.g. AISHELL-2) are 32 kHz.
    cuts = cuts.resample(16000)
    return cuts


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DataParallel or DistributedDataParallel if present."""
    if isinstance(module, (torch.nn.DataParallel, DDP)):
        return module.module
    return module


def _tower_trainable_params(tower: torch.nn.Module) -> list:
    """Adapter + attention-pool weights for one tower."""
    t = _unwrap(tower)
    return list(t.adapter.parameters()) + list(t.pool.parameters())


def _save_adapter(audio_tower, text_tower, embed_dim: int, path: Path):
    aw = _unwrap(audio_tower)
    ckpt: dict = {
        "audio_adapter": aw.adapter.state_dict(),
        "text_adapter": _unwrap(text_tower).adapter.state_dict(),
        "audio_pool": aw.pool.state_dict(),
        "text_pool": _unwrap(text_tower).pool.state_dict(),
        "embed_dim": embed_dim,
    }
    torch.save(ckpt, path)
    logger.info("Saved adapter checkpoint → %s", path)


def _load_adapter(audio_tower, text_tower, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    aw = _unwrap(audio_tower)
    aw.adapter.load_state_dict(ckpt["audio_adapter"])
    _unwrap(text_tower).adapter.load_state_dict(ckpt["text_adapter"])
    if "audio_pool" in ckpt:
        aw.pool.load_state_dict(ckpt["audio_pool"])
    tw = _unwrap(text_tower)
    if "text_pool" in ckpt:
        tw.pool.load_state_dict(ckpt["text_pool"])
    logger.info("Loaded adapter checkpoint ← %s", path)


# ---------------------------------------------------------------------------
# Evaluation: Recall@K
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_recall_at_k(
    audio_tower: nn.Module,
    text_tower: nn.Module,
    loader: DataLoader,
    vocab_or_by_lang,
    tokenizer,
    ks: list[int],
    device: torch.device,
    batch_text: int = 512,
) -> dict[int, float]:
    """Compute Recall@K on a validation split.

    ``vocab_or_by_lang`` is either a flat ``list[str]`` (legacy) or a
    ``dict`` with keys ``en`` / ``zh`` for language-specific retrieval pools.
    """
    audio_tower.eval()
    text_tower.eval()

    from rag_asr.dataset import tokenise_words

    bilingual = isinstance(vocab_or_by_lang, dict) and "all" not in vocab_or_by_lang
    if bilingual:
        vocab_by_lang = vocab_or_by_lang
        vocab_embs_by_lang = {}
        for lang, vocab in vocab_by_lang.items():
            logger.info("Pre-encoding %s val vocab (%d words)…", lang, len(vocab))
            parts = []
            for i in range(0, len(vocab), batch_text):
                chunk = vocab[i : i + batch_text]
                ids, mask = tokenise_words(chunk, tokenizer, device=device)
                parts.append(text_tower(ids, mask).cpu())
            vocab_embs_by_lang[lang] = torch.cat(parts, dim=0)
    else:
        vocab = vocab_or_by_lang["all"] if isinstance(vocab_or_by_lang, dict) else vocab_or_by_lang
        logger.info("Pre-encoding vocab (%d words)…", len(vocab))
        vocab_embs_parts = []
        for i in range(0, len(vocab), batch_text):
            chunk = vocab[i : i + batch_text]
            ids, mask = tokenise_words(chunk, tokenizer, device=device)
            vocab_embs_parts.append(text_tower(ids, mask).cpu())
        vocab_embs = torch.cat(vocab_embs_parts, dim=0)

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        feature_lens = batch["feature_lens"].to(device)
        hotwords_batch = batch["hotwords"]
        langs_batch = batch.get("langs")

        a_embs = audio_tower(features, feature_lens).cpu()

        for i, gt_hotwords in enumerate(hotwords_batch):
            if not gt_hotwords:
                continue
            if bilingual:
                lang = langs_batch[i]
                vocab = vocab_by_lang[lang]
                vocab_embs = vocab_embs_by_lang[lang]
            gt_set = {h.lower() for h in gt_hotwords}
            sims = a_embs[i] @ vocab_embs.T
            topk_idx = sims.topk(min(max_k, sims.shape[0])).indices.tolist()
            for k in ks:
                topk_k = {vocab[j].lower() for j in topk_idx[:k]}
                hits[k] += len(gt_set & topk_k) / len(gt_set)
            total += 1

    recall = {k: (hits[k] / total if total > 0 else 0.0) for k in ks}
    audio_tower.train()
    text_tower.train()
    return recall


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    # ------------------------------------------------------------------
    # DDP initialisation — torchrun sets LOCAL_RANK / RANK / WORLD_SIZE
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device  = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    # Different seed per rank for data diversity (sampler shuffle seeds differ)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # wait for rank-0 to create the directory

    _log_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if is_main:
        _log_handlers.append(logging.FileHandler(output_dir / "train.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=_log_handlers,
    )
    if is_main:
        logger.info("Args: %s", json.dumps(vars(args), indent=2))
    logger.info("DDP rank %d/%d on cuda:%d", rank, world_size, local_rank)

    # ------------------------------------------------------------------
    # 1. Build training CutSet and dataset
    # ------------------------------------------------------------------
    from lhotse.features import WhisperFbank, WhisperFbankConfig
    from rag_asr.dataset import (
        HotwordsRetrievalDataset,
        retrieval_collate_fn_factory,
        tokenise_words,
    )

    logger.info("Loading training manifests…")
    train_cuts = _load_cutset(args.supervisions, args.recordings)

    train_vocab_lists = _load_train_vocab_lists(args, train_cuts)
    val_vocab_lists = _load_val_vocab_lists(args, train_vocab_lists)
    bilingual_vocab = "en" in train_vocab_lists and "zh" in train_vocab_lists
    if bilingual_vocab:
        logger.info(
            "Bilingual train vocabs: en=%d words, zh=%d words",
            len(train_vocab_lists["en"]), len(train_vocab_lists["zh"]),
        )
    elif "all" in train_vocab_lists:
        logger.info("Single train vocab: %d words", len(train_vocab_lists["all"]))
    if "en" in val_vocab_lists and "zh" in val_vocab_lists:
        logger.info(
            "Bilingual val vocabs: en=%d words, zh=%d words",
            len(val_vocab_lists["en"]), len(val_vocab_lists["zh"]),
        )
    elif "all" in val_vocab_lists:
        logger.info("Single val vocab: %d words", len(val_vocab_lists["all"]))

    train_dataset = HotwordsRetrievalDataset(
        train_cuts, max_duration_s=args.max_duration_s
    )
    fbank_extractor = WhisperFbank(WhisperFbankConfig(num_filters=args.num_mel_bins))
    collate_fn = retrieval_collate_fn_factory(fbank_extractor)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Validation
    val_loader = None
    if args.val_supervisions:
        logger.info("Loading validation manifests…")
        val_cuts = _load_cutset(args.val_supervisions, args.val_recordings)
        val_dataset = HotwordsRetrievalDataset(
            val_cuts, max_duration_s=args.max_duration_s
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # 2. Load base model (frozen), build towers
    # ------------------------------------------------------------------
    logger.info("Loading base model from %s…", args.base_model_path)
    from rag_asr.dual_tower import (
        AmphionAudioTower,
        AmphionTextTower,
        per_positive_infonce_loss,
    )
    from rag_asr.model_loader import load_base_model, load_tokenizer

    base_model = load_base_model(args.base_model_path, device, dtype=torch.float16)
    tokenizer = load_tokenizer(args.base_model_path)

    audio_tower = AmphionAudioTower(
        base_model,
        embed_dim=args.embed_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    text_tower = AmphionTextTower(
        base_model,
        embed_dim=args.embed_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Resume from checkpoint if requested
    if args.resume:
        _load_adapter(audio_tower, text_tower, Path(args.resume))

    # Wrap towers in DDP.  Frozen backbone parameters (requires_grad=False)
    # are excluded from gradient reduction automatically; only the lightweight
    # adapter weights are synced via AllReduce across ranks.
    audio_tower = DDP(audio_tower, device_ids=[local_rank], find_unused_parameters=False)
    text_tower  = DDP(text_tower,  device_ids=[local_rank], find_unused_parameters=False)

    # Logit scale (= 1/τ), following CLAP / CLIP convention.
    # Either a fixed tensor or a learnable nn.Parameter depending on --learnable-temperature.
    _log_scale_val = torch.tensor(math.log(1.0 / args.temperature), dtype=torch.float32, device=device)
    if args.learnable_temperature:
        log_scale = torch.nn.Parameter(_log_scale_val)
    else:
        log_scale = _log_scale_val

    n_params = sum(p.numel() for p in _tower_trainable_params(audio_tower)) + \
               sum(p.numel() for p in _tower_trainable_params(text_tower))
    if is_main:
        logger.info("DDP enabled: world_size=%d", world_size)
        logger.info("Trainable adapter parameters: %s", f"{n_params:,}")
        logger.info("Initial temperature: %.4f (log_scale=%.4f, effective_tau=%.4f, learnable=%s)",
                    args.temperature, log_scale.item(), 1.0 / log_scale.exp().item(),
                    args.learnable_temperature)
        logger.info("Loss weights: w_a2t=%.2f  w_t2a=%.2f", args.loss_w_a2t, args.loss_w_t2a)

    # ------------------------------------------------------------------
    # 3. Optimiser & scheduler
    # ------------------------------------------------------------------
    trainable_params = (
        _tower_trainable_params(audio_tower)
        + _tower_trainable_params(text_tower)
        + ([log_scale] if args.learnable_temperature else [])
    )
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs

    # Warmup steps: explicit --warmup-steps takes priority over --warmup-ratio
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    if is_main:
        logger.info(
            "LR schedule: linear warmup for %d steps, then cosine decay to lr*0.01 "
            "over %d total steps  (warmup_ratio=%.3f)",
            warmup_steps, total_steps, warmup_steps / total_steps,
        )

    def _lr_lambda(current_step: int) -> float:
        """Linear warmup then cosine decay."""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        import math as _math
        cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        # Decay to 1% of peak lr
        return max(0.01, cosine)

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 3b. Pre-tokenise the training vocabulary (one-time cost at startup)
    #     Stores token IDs on CPU; during training we index into this table
    #     instead of calling the tokenizer every step, which eliminates the
    #     ~2–3 s/step CPU tokenization bottleneck.
    # ------------------------------------------------------------------
    _PRETOK_MAX_LEN = 20
    train_vocab_data: dict[str, dict] = {}
    for key, words in train_vocab_lists.items():
        if is_main:
            logger.info("Pre-tokenising %s train vocab (%d words)…", key, len(words))
        train_vocab_data[key] = _pretokenize_vocab(words, tokenizer, _PRETOK_MAX_LEN)
        if is_main:
            logger.info(
                "  %s pretok shape=%s",
                key, list(train_vocab_data[key]["ids_cpu"].shape),
            )

    val_vocab_for_eval = val_vocab_lists

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_k = min(args.val_recall_k)
    global_step = 0

    # Normalise validation trigger flags.
    # If both are 0, fall back to validating once per epoch (legacy behaviour).
    val_every_steps = args.val_every_steps  # 0 = disabled
    val_every_epoch = args.val_every_epoch  # 0 = disabled
    if val_every_steps == 0 and val_every_epoch == 0:
        val_every_epoch = 1  # backward-compatible default

    def _run_validation(tag: str):
        """Run validation on rank-0 only. Call only after dist.barrier()."""
        nonlocal best_val_loss
        if val_loader is None:
            return
        recall = evaluate_recall_at_k(
            _unwrap(audio_tower), _unwrap(text_tower), val_loader,
            val_vocab_for_eval, tokenizer, args.val_recall_k, device,
        )
        msg = "  ".join(f"R@{k}={v:.4f}" for k, v in sorted(recall.items()))
        logger.info("Validation [%s]: %s", tag, msg)

        primary_metric = recall.get(best_k, 0.0)
        if primary_metric > (1.0 - best_val_loss):
            best_val_loss = 1.0 - primary_metric
            logger.info(
                "  → New best R@%d=%.4f, saving best_adapter.pt", best_k, primary_metric,
            )
            _save_adapter(
                audio_tower, text_tower, args.embed_dim, output_dir / "best_adapter.pt",
            )

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # ensure different shuffle each epoch
        audio_tower.train()
        text_tower.train()

        epoch_loss = 0.0   # cumulative for epoch-end summary only
        epoch_pairs = 0
        window_loss = 0.0  # rolling window for step-level logging
        window_a2t  = 0.0
        window_t2a  = 0.0
        window_pairs = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            features = batch["features"].to(device)          # (B, T, F)
            feature_lens = batch["feature_lens"].to(device)  # (B,)
            hotwords_batch: list[list[str]] = batch["hotwords"]
            batch_langs: list[str] = batch.get(
                "langs", ["zh"] * len(hotwords_batch)
            )

            all_pos_words: list[str] = []
            valid_batch_idx: list[int] = []
            pair_langs: list[str] = []
            for i, hw_list in enumerate(hotwords_batch):
                for hw in hw_list:
                    all_pos_words.append(hw)
                    valid_batch_idx.append(i)
                    pair_langs.append(batch_langs[i])

            if not valid_batch_idx:
                continue

            pos_ids, pos_mask = tokenise_words(all_pos_words, tokenizer, device=device)
            t2a_labels = torch.tensor(valid_batch_idx, dtype=torch.long, device=device)
            valid_batch_idx_t = torch.tensor(valid_batch_idx, dtype=torch.long, device=device)

            def _compute_loss_a2t(a_embs_unique, p_embs, logit_scale):
                if bilingual_vocab:
                    loss_sum = 0.0
                    count = 0
                    for lang in ("en", "zh"):
                        pair_idx = [k for k, lg in enumerate(pair_langs) if lg == lang]
                        if not pair_idx:
                            continue
                        vd = train_vocab_data[lang]
                        exclude = {
                            w
                            for i, hws in enumerate(hotwords_batch)
                            if batch_langs[i] == lang
                            for w in hws
                        }
                        sampled = _sample_negative_indices(
                            vd, exclude, args.num_negatives - 1,
                        )
                        if not sampled:
                            continue
                        neg_ids = vd["ids_cpu"][sampled].to(device)
                        neg_mask = vd["mask_cpu"][sampled].to(device)
                        neg_embs = text_tower(neg_ids, neg_mask)
                        idx_t = torch.tensor(pair_idx, dtype=torch.long, device=device)
                        batch_idx_t = torch.tensor(
                            [valid_batch_idx[k] for k in pair_idx],
                            dtype=torch.long, device=device,
                        )
                        sub = per_positive_infonce_loss(
                            a_embs_unique[batch_idx_t], p_embs[idx_t], neg_embs, logit_scale,
                        )
                        loss_sum = loss_sum + sub * len(pair_idx)
                        count += len(pair_idx)
                    if count == 0:
                        return p_embs.new_zeros(())
                    return loss_sum / count

                vd = train_vocab_data["all"]
                exclude = {w for hws in hotwords_batch for w in hws}
                sampled = _sample_negative_indices(
                    vd, exclude, args.num_negatives - 1,
                )
                if not sampled:
                    return p_embs.new_zeros(())
                neg_ids = vd["ids_cpu"][sampled].to(device)
                neg_mask = vd["mask_cpu"][sampled].to(device)
                neg_embs = text_tower(neg_ids, neg_mask)
                return per_positive_infonce_loss(
                    a_embs_unique[valid_batch_idx_t],
                    p_embs,
                    neg_embs,
                    logit_scale,
                )

            def _forward():
                logit_scale = log_scale.exp()
                a_embs_unique = audio_tower(features, feature_lens)
                p_embs = text_tower(pos_ids, pos_mask)
                loss_a2t = _compute_loss_a2t(a_embs_unique, p_embs, logit_scale)
                sims_t2a = logit_scale * (p_embs @ a_embs_unique.T)
                loss_t2a = F.cross_entropy(sims_t2a, t2a_labels)
                total = args.loss_w_a2t * loss_a2t + args.loss_w_t2a * loss_t2a
                return total, loss_a2t.detach(), loss_t2a.detach()

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, loss_a2t_val, loss_t2a_val = _forward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_a2t_val, loss_t2a_val = _forward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

            # Clamp learnable log_scale to [0, log(100)] → τ ∈ [0.01, 1.0].
            if args.learnable_temperature:
                with torch.no_grad():
                    log_scale.clamp_(0.0, math.log(100))

            scheduler.step()
            global_step += 1
            n_pairs = len(valid_batch_idx)
            epoch_loss += loss.item() * n_pairs
            epoch_pairs += n_pairs
            window_loss += loss.item() * n_pairs
            window_pairs += n_pairs
            window_a2t  += loss_a2t_val.item() * n_pairs
            window_t2a  += loss_t2a_val.item() * n_pairs

            if is_main and global_step % args.log_every == 0:
                elapsed = time.time() - t0
                avg      = window_loss / max(window_pairs, 1)
                avg_a2t  = window_a2t  / max(window_pairs, 1)
                avg_t2a  = window_t2a  / max(window_pairs, 1)
                window_loss = window_a2t = window_t2a = 0.0
                window_pairs = 0
                lr_now = scheduler.get_last_lr()[0]
                temp_now = 1.0 / log_scale.exp().item()
                logger.info(
                    "epoch %d  step %d/%d  global_step %d  "
                    "loss=%.4f  a2t=%.4f  t2a=%.4f  lr=%.2e  temp=%.4f  t=%.1fs",
                    epoch, step, len(train_loader), global_step,
                    avg, avg_a2t, avg_t2a, lr_now, temp_now, elapsed,
                )

            # Step-level validation: barrier ensures all ranks finish the step
            # before rank-0 runs eval; second barrier resumes training together.
            if val_every_steps > 0 and global_step % val_every_steps == 0:
                dist.barrier()
                if is_main:
                    _save_adapter(
                        audio_tower, text_tower, args.embed_dim, output_dir / "last_adapter.pt"
                    )
                    _run_validation(f"step {global_step}")
                dist.barrier()
                audio_tower.train()
                text_tower.train()

        epoch_avg_loss = epoch_loss / max(epoch_pairs, 1)
        if is_main:
            logger.info(
                "=== Epoch %d finished: avg_loss=%.4f  elapsed=%.1fs ===",
                epoch, epoch_avg_loss, time.time() - t0,
            )

        dist.barrier()
        if is_main:
            # Always save latest checkpoint at epoch end
            _save_adapter(audio_tower, text_tower, args.embed_dim, output_dir / "last_adapter.pt")

            # Epoch-level validation
            if val_every_epoch > 0 and epoch % val_every_epoch == 0:
                _run_validation(f"epoch {epoch}")
            elif val_loader is None and epoch == 1:
                # No validation set: treat first checkpoint as best
                _save_adapter(
                    audio_tower, text_tower, args.embed_dim, output_dir / "best_adapter.pt"
                )
        dist.barrier()

    if is_main:
        logger.info("Training complete.  Outputs in %s", output_dir)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_entry():
    args = _build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main_entry()
