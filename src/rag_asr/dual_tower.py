"""Dual-tower retrieval model built on top of Amphion-4B components.

Architecture
------------
Audio tower
  frozen(audio_encoder → multi_modal_projector) → MLP adapter → attention pool → L2-normed embedding

Text tower
  frozen(language_model.embed_tokens) → MLP adapter → attention pool → L2-normed embedding

Training objective
  Per-positive InfoNCE (contrastive loss).  For each utterance with
  |P| ground-truth hotwords, we treat every (audio, hotword_p) pair as an
  independent anchor-positive example.  All pairs share the same random
  pool of global negatives (drawn once per batch, excluding all hotwords
  that appear anywhere in the batch to avoid false negatives).

Usage
-----
>>> audio_tower = AmphionAudioTower(base_model, embed_dim=512)
>>> text_tower  = AmphionTextTower(base_model, embed_dim=512)
>>> loss = per_positive_infonce_loss(a_embs, p_embs, n_embs, temperature=0.07)

Notes
-----
* ``base_model`` is a ``AmphionASRForCausalLM`` (HF) or any object that
  exposes ``.audio_encoder``, ``.multi_modal_projector``, and
  ``.language_model.model.embed_tokens``.  All three sub-modules are
  frozen inside the towers; only the adapter weights are trained.
* The projector's downsampled output (shape ``(B, T', llm_dim)``) is
  pooled with a mask derived from ``encoder_lens`` divided by the
  projector's downsampling rate.  This mirrors ``_encode()`` in
  ``modeling_amphion_asr.py`` faithfully.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.distributed
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: length-masked mean pooling
# ---------------------------------------------------------------------------

def masked_mean_pool(
    hidden: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool ``hidden`` over valid time steps defined by ``lengths``.

    Parameters
    ----------
    hidden : ``(B, T, D)``
    lengths : ``(B,)`` integer tensor — number of valid frames per item

    Returns
    -------
    ``(B, D)`` pooled representation
    """
    B, T, D = hidden.shape
    mask = torch.arange(T, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
    mask = mask.unsqueeze(2).float()  # (B, T, 1)
    pooled = (hidden * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()  # (B, D)
    return pooled


# ---------------------------------------------------------------------------
# Attention-weighted pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Learned attention-weighted pooling over the time axis.

    A single linear layer (no bias) maps each frame vector to a scalar
    attention score.  A masked softmax normalises the scores over valid
    frames so that padding is ignored, then the output is the weighted
    sum of the frame vectors.

    At initialisation the weights are near-zero, so the layer behaves
    approximately like mean pooling and introduces no loss spike.

    Parameters
    ----------
    dim : int
        Dimensionality of the input frame vectors.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden  : ``(B, T, D)``
        lengths : ``(B,)`` valid frame counts (clamp ≥ 1 before calling)

        Returns
        -------
        ``(B, D)`` attention-weighted sum
        """
        B, T, _ = hidden.shape
        scores = self.score(hidden).squeeze(-1)  # (B, T)
        pad_mask = torch.arange(T, device=hidden.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=1)           # (B, T)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


# ---------------------------------------------------------------------------
# Shared MLP adapter (2-layer with GELU)
# ---------------------------------------------------------------------------

class MLPAdapter(nn.Module):
    """Two-layer MLP with GELU: linear → GELU → linear.

    Projects from ``in_dim`` to ``embed_dim``.  Optionally uses an
    intermediate hidden dimension; defaults to ``max(in_dim, embed_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Audio Tower
# ---------------------------------------------------------------------------

def _detect_backend(base_model: nn.Module) -> str:
    """Return ``"amphion"`` or ``"qwen3"`` based on base_model config.

    ``amphion`` = ``AmphionASRForConditionalGeneration`` (4B, custom arch).
    ``qwen3``   = ``Qwen3ASRForConditionalGeneration`` (1.7B, Qwen3 arch).
    Falls back to ``"amphion"`` for unknown types.
    """
    model_type = getattr(getattr(base_model, "config", None), "model_type", "") or ""
    if "qwen3" in model_type.lower():
        return "qwen3"
    return "amphion"


class AmphionAudioTower(nn.Module):
    """Audio encoder branch of the dual-tower retrieval model.

    Supports two backends selected automatically from ``base_model.config.model_type``:

    * ``amphion`` (``AmphionASRForConditionalGeneration``, 4B):
      frozen ``audio_encoder`` + ``multi_modal_projector``; batched ``(B, T, F)``
      forward; adapter applied per-frame then pooled.

    * ``qwen3`` (``Qwen3ASRForConditionalGeneration``, 1.7B):
      frozen ``thinker.audio_tower`` (encoder + projector in one module);
      per-utterance loop because the Qwen3 audio_tower uses packed sequences
      internally; adapter applied to the packed frame output then pooled.

    Parameters
    ----------
    base_model
        The full ASR model (AmphionASR or Qwen3ASR).
    embed_dim : int
        Shared embedding dimension (same for audio and text towers).
    adapter_hidden_dim : optional int
        Hidden width of the MLP adapter.
    dropout : float
        Dropout rate inside the adapter.
    """

    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int = 512,
        adapter_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._backend = _detect_backend(base_model)

        if self._backend == "qwen3":
            # Qwen3ASR: encoder + projector are a single audio_tower module
            self.audio_tower = base_model.thinker.audio_tower
            for p in self.audio_tower.parameters():
                p.requires_grad_(False)
            _params = list(self.audio_tower.parameters())
            self._encoder_dtype: torch.dtype = _params[0].dtype if _params else torch.float16
            proj_out_dim = self.audio_tower.proj2.out_features
            self.downsample_rate: int = 1  # not used in qwen3 path
        else:
            # AmphionASR: separate audio_encoder + multi_modal_projector
            self.audio_encoder = base_model.audio_encoder
            self.projector = base_model.multi_modal_projector
            for p in self.audio_encoder.parameters():
                p.requires_grad_(False)
            for p in self.projector.parameters():
                p.requires_grad_(False)
            _enc_params = list(self.audio_encoder.parameters())
            _enc_bufs   = list(self.audio_encoder.buffers())
            if _enc_params:
                self._encoder_dtype = _enc_params[0].dtype
            elif _enc_bufs:
                self._encoder_dtype = _enc_bufs[0].dtype
            else:
                self._encoder_dtype = torch.float16
            proj_out_dim = self._infer_proj_out_dim(base_model)
            self.downsample_rate = getattr(self.projector, "downsample_rate", 1)

        self.adapter = MLPAdapter(
            in_dim=proj_out_dim,
            embed_dim=embed_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=dropout,
        )
        self.pool = AttentionPool(embed_dim)
        self.embed_dim = embed_dim

    @staticmethod
    def _infer_proj_out_dim(base_model: nn.Module) -> int:
        """Return the output dimension of AmphionASR's multi_modal_projector."""
        cfg = getattr(base_model, "config", None)
        if cfg is not None:
            proj_cfg = getattr(cfg, "projector_config", None)
            if proj_cfg is not None:
                return getattr(proj_cfg, "llm_dim", 2048)
        # Fallback: inspect the last linear layer of the projector
        proj = base_model.multi_modal_projector
        for m in reversed(list(proj.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
        raise RuntimeError(
            "Cannot infer projector output dimension from base_model. "
            "Please pass adapter_in_dim explicitly."
        )

    def _forward_amphion(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """AmphionASR batched forward: encoder → projector → adapter → pool."""
        with torch.no_grad():
            feats = features.to(dtype=self._encoder_dtype)
            max_len = int(feature_lens.max().item())
            feats = feats[:, :max_len, :].contiguous()
            enc_out, enc_lens = self.audio_encoder(feats, feature_lens)
            proj_out = self.projector(enc_out)          # (B, T_proj, llm_dim)
            proj_lens = (enc_lens // self.downsample_rate).clamp(min=1)

        proj_f = proj_out.float()
        B, T, proj_dim = proj_f.shape
        embs = self.adapter(proj_f.view(B * T, proj_dim)).view(B, T, self.embed_dim)
        pooled = self.pool(embs, proj_lens)
        return F.normalize(pooled, dim=-1)

    def _forward_qwen3(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Qwen3ASR per-utterance forward: audio_tower → adapter → pool.

        Qwen3's audio_tower uses packed sequences internally, so we call it
        one utterance at a time (matching how Qwen3ASRThinker calls it) and
        stack the pooled results.
        """
        pooled_list: list[torch.Tensor] = []
        for i in range(features.shape[0]):
            T_i = int(feature_lens[i].item())
            # audio_tower expects (F, T) for a single utterance
            feat_i = features[i, :T_i, :].to(dtype=self._encoder_dtype).T.contiguous()
            fl_i = feature_lens[i:i + 1]  # (1,)
            with torch.no_grad():
                out = self.audio_tower(feat_i, feature_lens=fl_i)
            # last_hidden_state: (T'_i, proj_dim) packed frames for one utt
            hidden = out.last_hidden_state.float()      # (T'_i, proj_dim)
            embs_i = self.adapter(hidden)               # (T'_i, embed_dim)
            lengths_i = torch.tensor(
                [embs_i.shape[0]], dtype=torch.long, device=embs_i.device
            )
            pooled_i = self.pool(embs_i.unsqueeze(0), lengths_i).squeeze(0)
            pooled_list.append(F.normalize(pooled_i, dim=-1))
        return torch.stack(pooled_list)                 # (B, embed_dim)

    def _projector_dim_amphion(self) -> int:
        for m in reversed(list(self.projector.modules())):
            if isinstance(m, nn.Linear):
                return int(m.out_features)
        raise RuntimeError("cannot infer projector output dim")

    @property
    def projector_dim(self) -> int:
        """Output dimension of encoder → projector (before retrieval adapter)."""
        if self._backend == "qwen3":
            return int(self.audio_tower.proj2.out_features)
        return self._projector_dim_amphion()

    def forward_with_projector(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode audio and return retrieval embedding plus projector frames.

        Returns
        -------
        pooled : ``(B, embed_dim)`` L2-normalised retrieval embedding
        proj : ``(B, T', D_proj)`` float32 projector output (padded)
        proj_lens : ``(B,)`` valid projector frame counts
        """
        if self._backend == "qwen3":
            return self._forward_with_projector_qwen3(features, feature_lens)
        return self._forward_with_projector_amphion(features, feature_lens)

    def _forward_with_projector_amphion(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feats = features.to(dtype=self._encoder_dtype)
            max_len = int(feature_lens.max().item())
            feats = feats[:, :max_len, :].contiguous()
            enc_out, enc_lens = self.audio_encoder(feats, feature_lens)
            proj_out = self.projector(enc_out)
            proj_lens = (enc_lens // self.downsample_rate).clamp(min=1)

        proj_f = proj_out.float()
        B, T, proj_dim = proj_f.shape
        embs = self.adapter(proj_f.view(B * T, proj_dim)).view(B, T, self.embed_dim)
        pooled = self.pool(embs, proj_lens)
        return F.normalize(pooled, dim=-1), proj_f, proj_lens

    def _forward_with_projector_qwen3(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_list: list[torch.Tensor] = []
        proj_list: list[torch.Tensor] = []
        lens_list: list[int] = []
        for i in range(features.shape[0]):
            T_i = int(feature_lens[i].item())
            feat_i = features[i, :T_i, :].to(dtype=self._encoder_dtype).T.contiguous()
            fl_i = feature_lens[i : i + 1]
            with torch.no_grad():
                out = self.audio_tower(feat_i, feature_lens=fl_i)
            hidden = out.last_hidden_state.float()
            embs_i = self.adapter(hidden)
            lengths_i = torch.tensor(
                [embs_i.shape[0]], dtype=torch.long, device=embs_i.device
            )
            pooled_i = self.pool(embs_i.unsqueeze(0), lengths_i).squeeze(0)
            pooled_list.append(F.normalize(pooled_i, dim=-1))
            proj_list.append(hidden)
            lens_list.append(hidden.shape[0])

        max_t = max(lens_list)
        D = proj_list[0].shape[1]
        device = proj_list[0].device
        proj = torch.zeros(len(proj_list), max_t, D, dtype=torch.float32, device=device)
        for i, h in enumerate(proj_list):
            proj[i, : lens_list[i], :] = h
        proj_lens = torch.tensor(lens_list, dtype=torch.long, device=device)
        return torch.stack(pooled_list), proj, proj_lens

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : ``(B, T, F)`` float — mel-filterbank features
        feature_lens : ``(B,)`` long — valid frame counts

        Returns
        -------
        ``(B, embed_dim)`` L2-normalised embedding
        """
        if self._backend == "qwen3":
            return self._forward_qwen3(features, feature_lens)
        return self._forward_amphion(features, feature_lens)


# ---------------------------------------------------------------------------
# Text Tower
# ---------------------------------------------------------------------------

class AmphionTextTower(nn.Module):
    """Text encoder branch of the dual-tower retrieval model.

    Uses the frozen token-embedding table from the base language model,
    followed by attention-weighted pooling and a trainable MLP adapter.

    Parameters
    ----------
    base_model
        The full ``AmphionASRForCausalLM``.  ``language_model.model.embed_tokens``
        is extracted and frozen.
    embed_dim : int
        Shared embedding dimension.
    adapter_hidden_dim : optional int
        Hidden width of the MLP adapter.
    dropout : float
    """

    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int = 512,
        adapter_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Extract and freeze embed_tokens — path differs by backend
        _backend = _detect_backend(base_model)
        if _backend == "qwen3":
            self.embed_tokens: nn.Embedding = base_model.thinker.model.embed_tokens
        else:
            self.embed_tokens = base_model.language_model.model.embed_tokens
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)

        token_dim = self.embed_tokens.embedding_dim

        # Trainable adapter
        self.adapter = MLPAdapter(
            in_dim=token_dim,
            embed_dim=embed_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=dropout,
        )
        self.pool = AttentionPool(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``(N, L)`` long
        attention_mask : ``(N, L)`` float/bool — 1 for valid tokens

        Returns
        -------
        ``(N, embed_dim)`` L2-normalised embedding
        """
        with torch.no_grad():
            token_embs = self.embed_tokens(input_ids)  # (N, L, token_dim)

        # Apply adapter per-token, then pool in embed_dim space.
        N, L, token_dim = token_embs.shape
        embs = self.adapter(token_embs.float().view(N * L, token_dim)).view(N, L, self.embed_dim)
        lengths = attention_mask.sum(dim=1).long()    # (N,)
        pooled = self.pool(embs, lengths)

        return F.normalize(pooled, dim=-1)


# ---------------------------------------------------------------------------
# Per-positive InfoNCE loss
# ---------------------------------------------------------------------------

def per_positive_infonce_loss(
    audio_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    pos_counts: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Per-positive InfoNCE contrastive loss.

    For every utterance ``i`` with ``pos_counts[i]`` positive hotwords, we
    compute one InfoNCE term per (audio_i, hotword_p) pair.  All terms in
    the batch share the same negative pool ``neg_embs``.  The final loss is
    the mean over all valid anchor-positive pairs.

    Parameters
    ----------
    audio_embs : ``(B, D)`` — one embedding per utterance
    pos_embs   : ``(P, D)`` — concatenated positive hotword embeddings
                  (P = sum of pos_counts)
    neg_embs   : ``(N, D)`` — shared negative embeddings
    pos_counts : ``(B,)`` long — number of positives per utterance.
                  Must satisfy ``sum(pos_counts) == P``.
    temperature : float

    Returns
    -------
    Scalar loss tensor (mean InfoNCE over all pairs, differentiable w.r.t.
    adapter parameters of both towers).
    """
    assert audio_embs.shape[0] == pos_counts.shape[0], (
        f"Batch size mismatch: audio_embs {audio_embs.shape[0]} vs "
        f"pos_counts {pos_counts.shape[0]}"
    )
    assert pos_embs.shape[0] == int(pos_counts.sum().item()), (
        f"pos_embs rows ({pos_embs.shape[0]}) must equal sum(pos_counts) "
        f"({pos_counts.sum().item()})"
    )

    inv_temp = 1.0 / temperature
    losses: list[torch.Tensor] = []
    pos_cursor = 0

    for i in range(audio_embs.shape[0]):
        n_pos = int(pos_counts[i].item())
        if n_pos == 0:
            continue
        a = audio_embs[i]  # (D,)
        positives = pos_embs[pos_cursor : pos_cursor + n_pos]  # (n_pos, D)
        pos_cursor += n_pos

        # Similarity scores
        pos_scores = (a.unsqueeze(0) * positives).sum(dim=-1) * inv_temp  # (n_pos,)
        neg_scores = (a.unsqueeze(0) * neg_embs).sum(dim=-1) * inv_temp  # (N,)

        # Denominator: all negatives + each positive independently
        for j in range(n_pos):
            # Numerator: this positive
            logit_pos = pos_scores[j]
            # Denominator: this positive + all negatives
            all_logits = torch.cat([logit_pos.unsqueeze(0), neg_scores], dim=0)  # (1+N,)
            log_softmax = F.log_softmax(all_logits, dim=0)
            losses.append(-log_softmax[0])

    if not losses:
        return audio_embs.sum() * 0.0  # zero loss, keeps grad graph alive

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# DDP all-gather helper (CLAP-style, gradient-preserving)
# ---------------------------------------------------------------------------

def gather_features(
    audio_embs: torch.Tensor,
    text_embs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather embeddings from all DDP ranks while preserving gradients.

    Uses ``torch.distributed.nn.all_gather`` (differentiable) so that every
    rank receives gradients for the full concatenated tensor.  When there is
    only one process (world_size == 1), returns the inputs unchanged.

    This expands the effective batch from ``B`` (per-rank) to ``B * world_size``,
    giving each anchor ``B * world_size - 1`` negatives instead of ``B - 1``.

    Parameters
    ----------
    audio_embs : ``(B, D)`` L2-normalised audio embeddings on the current rank
    text_embs  : ``(B, D)`` L2-normalised text embeddings on the current rank

    Returns
    -------
    all_audio : ``(B * world_size, D)``
    all_text  : ``(B * world_size, D)``
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return audio_embs, text_embs
    if torch.distributed.get_world_size() == 1:
        return audio_embs, text_embs

    all_audio = torch.cat(torch.distributed.nn.all_gather(audio_embs), dim=0)
    all_text  = torch.cat(torch.distributed.nn.all_gather(text_embs),  dim=0)
    return all_audio, all_text


# ---------------------------------------------------------------------------
# Symmetric InfoNCE loss (CLIP-style, in-batch + all-gather)
# ---------------------------------------------------------------------------

def symmetric_infonce_loss(
    audio_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    logit_scale: torch.Tensor,
    w_a2t: float = 1.0,
    w_t2a: float = 1.0,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style) with in-batch negatives only.

    Follows the CLAP / CLIP convention: ``logit_scale`` = 1/τ = exp(log_scale).

    Expects exactly **one positive text per audio** (pos_embs[i] ↔ audio_embs[i]).
    Both directions use the same B×B similarity matrix (text direction is its
    transpose), so they are always equally difficult — no asymmetry possible.

    Call ``gather_features`` before this function to expand B across DDP ranks.

    Parameters
    ----------
    audio_embs  : ``(B, D)`` L2-normalised audio embeddings
    pos_embs    : ``(B, D)`` L2-normalised positive text embeddings (one per audio)
    logit_scale : scalar tensor — exp(log_scale), i.e. 1/τ
    w_a2t       : weight for audio→text direction (default 1.0)
    w_t2a       : weight for text→audio direction (default 1.0)

    Returns
    -------
    Scalar loss.
    """
    labels = torch.arange(audio_embs.shape[0], device=audio_embs.device)

    logits = logit_scale * (audio_embs @ pos_embs.T)   # (B, B)
    loss_a2t = F.cross_entropy(logits,   labels)
    loss_t2a = F.cross_entropy(logits.T, labels)
    total = w_a2t * loss_a2t + w_t2a * loss_t2a
    return total / (w_a2t + w_t2a)


# ---------------------------------------------------------------------------
# Global-negative InfoNCE loss (per-audio N-way classification)
# ---------------------------------------------------------------------------

def per_positive_infonce_loss(
    audio_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE loss with a shared global negative pool.

    For each audio i, the classification task is:
        candidates = [ pos_embs[i],  neg_embs[0], …, neg_embs[N_neg-1] ]
        label      = 0   (positive is always at index 0)
        total N    = 1 + N_neg   (== ``--num-negatives`` in the training script)

    The negative pool is **shared across all audios in the batch**, so only
    ``N_neg`` extra text encodings are needed per step regardless of batch
    size.  This is the L_a2t direction only — L_t2a is not applicable because
    the global negatives carry no paired audio.

    Parameters
    ----------
    audio_embs  : ``(B, D)`` L2-normalised audio embeddings
    pos_embs    : ``(B, D)`` L2-normalised positive text embeddings
    neg_embs    : ``(N_neg, D)`` L2-normalised global negative text embeddings
    logit_scale : scalar tensor — exp(log_scale), i.e. 1/τ

    Returns
    -------
    Scalar loss.
    """
    B = audio_embs.shape[0]
    # Per-audio positive similarity: element-wise dot product → (B, 1)
    pos_sims = (audio_embs * pos_embs).sum(dim=-1, keepdim=True)
    # Each audio vs every global negative → (B, N_neg)
    neg_sims = audio_embs @ neg_embs.T
    # Concatenate: (B, 1 + N_neg)  label = 0 for all
    logits = logit_scale * torch.cat([pos_sims, neg_sims], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=audio_embs.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Convenience: build both towers from a loaded base model
# ---------------------------------------------------------------------------

def build_towers_from_base(
    base_model: nn.Module,
    embed_dim: int = 512,
    adapter_hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
) -> Tuple["AmphionAudioTower", "AmphionTextTower"]:
    """Construct audio and text towers sharing the same ``embed_dim``."""
    audio_tower = AmphionAudioTower(
        base_model,
        embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
        dropout=dropout,
    )
    text_tower = AmphionTextTower(
        base_model,
        embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
        dropout=dropout,
    )
    return audio_tower, text_tower
