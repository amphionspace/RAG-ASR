"""AmphionASR model for conditional generation (Speech-to-Text).

This module re-implements the ``Amphion_LLM`` architecture as a
``transformers.PreTrainedModel`` so that it works natively with the
HuggingFace ecosystem (``from_pretrained``, ``save_pretrained``,
PEFT, TRL, etc.).

Key design decisions
--------------------
* All audio encoder types are wrapped in a unified ``AudioEncoderWrapper``
  interface with a ``build_audio_encoder`` factory, so ``_encode()`` has
  no if/elif branching and new encoders can be added without touching
  existing code.
* ``SwooshR`` is re-implemented in pure PyTorch so that there is no
  dependency on the ``k2`` C++ library at inference time.
* Special-token IDs (``<speech>``, ``<start_text>``, …) are persisted
  in ``config.json`` and no longer patched onto the LLM config at
  runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_pt_utils import LabelSmoother

try:
    from .configuration_amphion_asr import AmphionASRConfig
except ImportError:
    from rag_asr.backends.amphion.configuration_amphion_asr import AmphionASRConfig

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ---------------------------------------------------------------------------
# Pure-PyTorch SwooshR (no k2 dependency)
# ---------------------------------------------------------------------------

class SwooshR(nn.Module):
    """``swoosh_r(x) = log(1 + exp(x − 1)) − 0.08 * x − 0.313261687``

    Pure PyTorch implementation of the activation used in Icefall /
    k2.  Numerically equivalent to ``k2.swoosh_r`` but does not require
    the k2 C++ extension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


# ---------------------------------------------------------------------------
# Unified AudioEncoderWrapper interface
# ---------------------------------------------------------------------------

class AudioEncoderWrapper(nn.Module):
    """Base class for all audio encoder wrappers.

    Provides a unified interface so that the model's ``_encode()`` method
    has zero branching logic regardless of the underlying encoder type.
    """

    def forward(
        self, features_btf: torch.Tensor, feature_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features_btf: (B, T, F) mel features.
            feature_lens:  (B,) frame counts per utterance.
        Returns:
            encoder_out:      (B, T', D) encoder output.
            encoder_out_lens: (B,) output lengths per utterance.
        """
        raise NotImplementedError

    @staticmethod
    def get_output_lengths(input_lengths: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Predict output token count from input frame count.

        Required by the vLLM prompt-replacement logic.
        """
        raise NotImplementedError


class Qwen3AudioEncoderWrapper(AudioEncoderWrapper):
    """Wrapper for encoder_type ``qwen3asr`` (Qwen3-AuT via ``qwen_asr``)."""

    def __init__(self, config_dict: dict):
        super().__init__()
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRAudioEncoder,
        )
        cfg = Qwen3ASRAudioEncoder.config_class(**config_dict)
        self.encoder = Qwen3ASRAudioEncoder(cfg)
        self._strip_proj()

    def _strip_proj(self):
        for attr in ("proj1", "act", "proj2"):
            if hasattr(self.encoder, attr):
                setattr(self.encoder, attr, nn.Identity())

    def forward(
        self, features_btf: torch.Tensor, feature_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs: List[torch.Tensor] = []
        lens: List[int] = []
        with torch.set_grad_enabled(self.encoder.training):
            for b in range(features_btf.shape[0]):
                t = min(max(int(feature_lens[b].item()), 0), features_btf.shape[1])
                feat_tf = features_btf[b, :t, :]
                t_len = feature_lens[b : b + 1].clamp(max=t)
                feat_ft = feat_tf.transpose(0, 1).contiguous()
                out = self.encoder(feat_ft, feature_lens=t_len)
                s = out.last_hidden_state
                seqs.append(s)
                lens.append(s.shape[0])
        encoder_lens = torch.tensor(lens, device=seqs[0].device, dtype=torch.long)
        encoder_outs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        return encoder_outs, encoder_lens

    @staticmethod
    def get_output_lengths(input_lengths):
        leave = input_lengths % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class OmniMoeAudioEncoderWrapper(AudioEncoderWrapper):
    """Wrapper for encoder_type ``qwen3omni_captioner`` and ``qwen3omni``."""

    def __init__(self, config_dict: dict):
        super().__init__()
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )
        cfg = Qwen3OmniMoeAudioEncoderConfig(**config_dict)
        self.encoder = Qwen3OmniMoeAudioEncoder(cfg)
        self._strip_proj()

    def _strip_proj(self):
        for attr in ("proj1", "act", "proj2"):
            if hasattr(self.encoder, attr):
                setattr(self.encoder, attr, nn.Identity())

    def forward(
        self, features_btf: torch.Tensor, feature_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs: List[torch.Tensor] = []
        lens: List[int] = []
        with torch.set_grad_enabled(self.encoder.training):
            for b in range(features_btf.shape[0]):
                t = min(max(int(feature_lens[b].item()), 0), features_btf.shape[1])
                feat_tf = features_btf[b, :t, :]
                t_len = feature_lens[b : b + 1].clamp(max=t)
                feat_ft = feat_tf.transpose(0, 1).contiguous()
                out = self.encoder(input_features=feat_ft, feature_lens=t_len)
                s = out.last_hidden_state
                seqs.append(s)
                lens.append(s.shape[0])
        encoder_lens = torch.tensor(lens, device=seqs[0].device, dtype=torch.long)
        encoder_outs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        return encoder_outs, encoder_lens

    @staticmethod
    def get_output_lengths(input_lengths):
        leave = input_lengths % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class ZipformerAudioEncoderWrapper(AudioEncoderWrapper):
    """Wrapper for encoder_type ``zipformer`` (Icefall Zipformer2)."""

    _MIN_FRAMES = 9

    def __init__(self, config_dict: dict):
        super().__init__()
        try:
            from .zipformer_inference import build_zipformer_encoder
        except ImportError:
            from rag_asr.backends.amphion.zipformer_inference import build_zipformer_encoder
        self.encoder = build_zipformer_encoder(config_dict)

    def forward(
        self, features_btf: torch.Tensor, feature_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        too_short = feature_lens < self._MIN_FRAMES
        if too_short.any():
            feature_lens = feature_lens.clone()
            feature_lens[too_short] = self._MIN_FRAMES
        encoder_outs, encoder_lens = self.encoder(features_btf, feature_lens)
        if too_short.any():
            encoder_lens = encoder_lens.clone()
            encoder_lens[too_short] = 0
        return encoder_outs, encoder_lens

    @staticmethod
    def get_output_lengths(input_lengths):
        return ((input_lengths - 7) // 2) // 2


# ---------------------------------------------------------------------------
# Encoder registry and factory
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY = {
    "qwen3asr": Qwen3AudioEncoderWrapper,
    "qwen3omni_captioner": OmniMoeAudioEncoderWrapper,
    "qwen3omni": OmniMoeAudioEncoderWrapper,
    "zipformer": ZipformerAudioEncoderWrapper,
}


def build_audio_encoder(config: AmphionASRConfig) -> AudioEncoderWrapper:
    """Construct the appropriate ``AudioEncoderWrapper`` from model config."""
    enc_cfg = config.audio_encoder_config
    enc_dict = enc_cfg.to_dict() if hasattr(enc_cfg, "to_dict") else dict(enc_cfg)
    enc_dict.pop("model_type", None)
    enc_dict.pop("transformers_version", None)
    cls = _ENCODER_REGISTRY.get(config.encoder_type)
    if cls is None:
        raise ValueError(
            f"Unsupported encoder_type: {config.encoder_type!r}. "
            f"Available: {sorted(_ENCODER_REGISTRY)}"
        )
    return cls(enc_dict)


# ---------------------------------------------------------------------------
# Multi-modal projector
# ---------------------------------------------------------------------------

class AmphionASRMultiModalProjector(nn.Module):
    """Projects encoder outputs to the LLM hidden dimension.

    Concatenates ``downsample_rate`` consecutive frames along the
    feature axis then applies ``Linear → SwooshR → Linear``.
    """

    def __init__(self, config: AmphionASRConfig):
        super().__init__()
        proj_cfg = config.projector_config
        self.downsample_rate = proj_cfg.downsample_rate
        self.proj = nn.Sequential(
            nn.Dropout(proj_cfg.dropout),
            nn.Linear(proj_cfg.encoder_dim * self.downsample_rate, proj_cfg.llm_dim),
            SwooshR(),
            nn.Linear(proj_cfg.llm_dim, proj_cfg.llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        x = x.contiguous().view(
            batch_size,
            seq_len // self.downsample_rate,
            feat_dim * self.downsample_rate,
        )
        return self.proj(x)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AmphionASRCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    accuracy: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

class AmphionASRPreTrainedModel(PreTrainedModel):
    config_class = AmphionASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AmphionASRMultiModalProjector"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

def _migrate_old_state_dict(state_dict: dict) -> dict:
    """Remap old-format keys ``audio_encoder.*`` (pre-wrapper) to the
    new ``audio_encoder.encoder.*`` layout.

    Old models stored encoder weights directly under ``audio_encoder.``
    (e.g. ``audio_encoder.conv1.weight``).  The wrapper layer inserts an
    extra ``.encoder.`` level.  This function transparently upgrades old
    checkpoints so that ``from_pretrained`` works without re-conversion.
    """
    new_state = {}
    migrated = 0
    for k, v in state_dict.items():
        if k.startswith("audio_encoder.") and not k.startswith("audio_encoder.encoder."):
            new_key = k.replace("audio_encoder.", "audio_encoder.encoder.", 1)
            new_state[new_key] = v
            migrated += 1
        else:
            new_state[k] = v
    if migrated > 0:
        logger.info(
            "Migrated %d state dict keys from old audio_encoder.* "
            "to audio_encoder.encoder.* format",
            migrated,
        )
    return new_state


class AmphionASRForConditionalGeneration(AmphionASRPreTrainedModel):
    """Amphion ASR model: audio encoder + projector + causal LM.

    Compatible with ``AutoModelForCausalLM.from_pretrained(...,
    trust_remote_code=True)`` once the config's ``auto_map`` is set.
    """

    def __init__(self, config: AmphionASRConfig):
        super().__init__(config)

        # --- Audio encoder (unified wrapper) ---------------------------------
        self.audio_encoder = build_audio_encoder(config)

        # --- Multi-modal projector -------------------------------------------
        self.multi_modal_projector = AmphionASRMultiModalProjector(config)

        # --- Language model ---------------------------------------------------
        text_cfg = config.text_config
        if isinstance(text_cfg, dict):
            from transformers import AutoConfig
            text_cfg = AutoConfig.for_model(**text_cfg)
        self.language_model = AutoModelForCausalLM.from_config(text_cfg)
        self._text_config = text_cfg

        # --- Prompt embedding (learnable vectors for special tokens) ----------
        hidden_size = (
            text_cfg.hidden_size
            if hasattr(text_cfg, "hidden_size")
            else config.projector_config.llm_dim
        )
        self.prompt_embedding = nn.Embedding(config.num_prompt_tokens, hidden_size)

        self.post_init()

    # ------------------------------------------------------------------
    # Backward-compatible state dict loading
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = _migrate_old_state_dict(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_btf(
        feature: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Ensure *feature* has shape ``(B, T, F)``."""
        if feature.ndim != 3:
            raise ValueError(
                f"Expected 3-D feature (B,T,F) or (B,F,T), got {tuple(feature.shape)}"
            )
        if feature.shape[-1] in (80, 128):
            return feature.contiguous()
        if feature.shape[1] in (80, 128):
            return feature.transpose(1, 2).contiguous()
        return feature.contiguous()

    def _encode(
        self,
        feature: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_lens = feature_lens.to(dtype=torch.long)
        feats_btf = self._normalize_to_btf(feature, feature_lens)
        max_len = feature_lens.max().item()
        feats_btf = feats_btf[:, :max_len, :].contiguous()
        encoder_dtype = next(self.audio_encoder.parameters()).dtype
        feats_btf = feats_btf.to(dtype=encoder_dtype)
        return self.audio_encoder(feats_btf, feature_lens)

    # ------------------------------------------------------------------
    # Speech / text merging
    # ------------------------------------------------------------------

    def _merge_input_ids_with_speech_features(
        self,
        speech_features: torch.Tensor,
        speech_feature_lens: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Replace ``<speech>`` placeholder(s) with encoder embeddings
        and inject learned prompt embeddings at boundary-token positions.

        Supports both single-audio (1 ``<speech>``) and multi-audio
        (N ``<speech>``) prompts.  Audio segments in *speech_features*
        are assigned to ``<speech>`` positions left-to-right across the
        batch via a running cursor — matching the Qwen2-Audio convention
        where all audio segments are concatenated along the batch dim.
        """
        cfg = self.config
        batch_size = input_ids.shape[0]
        audio_cursor = 0

        final_inputs_embeds = []
        final_attention_masks = []
        final_outputs_labels = []

        for i in range(batch_size):
            active = attention_mask[i].bool()
            active_ids = input_ids[i][active]
            active_embeds = inputs_embeds[i][active].clone()

            # Inject prompt embeddings at all boundary-token positions
            _prompt_map = [
                (cfg.start_text_token_id, 0),
                (cfg.end_text_token_id, 1),
                (cfg.start_speech_token_id, 2),
                (cfg.end_speech_token_id, 3),
            ]
            for token_id, weight_idx in _prompt_map:
                positions = (active_ids == token_id).nonzero(as_tuple=True)[0]
                for p in positions:
                    active_embeds[p] = self.prompt_embedding.weight[weight_idx].to(
                        active_embeds.dtype
                    )

            # Locate all <speech> positions (sorted left-to-right)
            speech_positions = torch.sort(
                (active_ids == cfg.default_speech_token_id).nonzero(as_tuple=True)[0]
            )[0]
            n_speech = speech_positions.size(0)
            assert n_speech >= 1, f"Expected >=1 <speech> tokens, got {n_speech}"

            # Splice: replace each <speech> with its audio features
            embed_parts: List[torch.Tensor] = []
            label_parts: Optional[List[torch.Tensor]] = [] if labels is not None else None
            active_labels = labels[i][active] if labels is not None else None
            current_start = 0

            for k in range(n_speech):
                pos = speech_positions[k].item()
                feat_len = int(speech_feature_lens[audio_cursor].item())

                embed_parts.append(active_embeds[current_start:pos])
                embed_parts.append(speech_features[audio_cursor, :feat_len])

                if label_parts is not None:
                    label_parts.append(active_labels[current_start:pos])
                    label_parts.append(torch.full(
                        (feat_len,), IGNORE_TOKEN_ID,
                        dtype=labels.dtype, device=labels.device,
                    ))

                current_start = pos + 1
                audio_cursor += 1

            embed_parts.append(active_embeds[current_start:])
            merged_embed = torch.cat(embed_parts, dim=0)

            merged_mask = torch.ones(
                merged_embed.size(0),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            final_inputs_embeds.append(merged_embed.flip(dims=[0]))
            final_attention_masks.append(merged_mask.flip(dims=[0]))

            if label_parts is not None:
                label_parts.append(active_labels[current_start:])
                merged_labels = torch.cat(label_parts, dim=0)
                final_outputs_labels.append(merged_labels.flip(dims=[0]))

        final_inputs_embeds = pad_sequence(
            final_inputs_embeds, batch_first=True, padding_value=0.0
        ).flip(dims=[1])
        final_attention_masks = pad_sequence(
            final_attention_masks, batch_first=True, padding_value=False
        ).flip(dims=[1])

        if labels is not None:
            final_outputs_labels = pad_sequence(
                final_outputs_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID
            ).flip(dims=[1])
        else:
            final_outputs_labels = None

        return final_inputs_embeds, final_attention_masks, final_outputs_labels

    # ------------------------------------------------------------------
    # forward / generate
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_lens: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, AmphionASRCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # When past_key_values are provided we are in autoregressive
        # decoding mode — the audio has already been encoded and merged
        # into the KV cache, so skip encoder + merge entirely.
        if past_key_values is not None or inputs_embeds is not None:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            if return_dict:
                return AmphionASRCausalLMOutput(
                    loss=outputs.loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=getattr(outputs, "hidden_states", None),
                    attentions=getattr(outputs, "attentions", None),
                )
            return outputs

        # --- Encode audio -----------------------------------------------------
        encoder_outs, encoder_lens = self._encode(input_features, feature_lens)

        # --- Project ----------------------------------------------------------
        speech_features = self.multi_modal_projector(encoder_outs)
        encoder_lens = encoder_lens // self.multi_modal_projector.downsample_rate

        # --- Merge speech + text embeddings -----------------------------------
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged_embeds, merged_mask, merged_labels = (
            self._merge_input_ids_with_speech_features(
                speech_features,
                encoder_lens,
                text_embeds,
                input_ids,
                attention_mask,
                labels,
            )
        )

        # --- LLM forward -----------------------------------------------------
        outputs = self.language_model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Compute token-level accuracy (non-differentiable metric)
        acc = None
        if merged_labels is not None:
            with torch.no_grad():
                preds = torch.argmax(outputs.logits, dim=-1)
                mask = merged_labels[:, 1:] != IGNORE_TOKEN_ID
                if mask.any():
                    correct = (preds[:, :-1][mask] == merged_labels[:, 1:][mask]).float()
                    acc = correct.mean()

        if return_dict:
            return AmphionASRCausalLMOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
                accuracy=acc,
            )
        return outputs

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Encode audio, merge with text embeddings, then delegate to
        the language model's ``generate``."""

        # Encode + project
        encoder_outs, encoder_lens = self._encode(input_features, feature_lens)
        speech_features = self.multi_modal_projector(encoder_outs)
        encoder_lens = encoder_lens // self.multi_modal_projector.downsample_rate

        # Merge
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged_embeds, merged_mask, _ = self._merge_input_ids_with_speech_features(
            speech_features, encoder_lens, text_embeds, input_ids, attention_mask
        )

        # Provide sane defaults from the LLM config
        lm_cfg = self.language_model.config
        _NON_GENERATE_KEYS = {
            "input_features", "feature_lens", "labels",
            "solution", "prompt_id", "request_id",
        }
        gen_kwargs = dict(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            max_new_tokens=kwargs.pop("max_new_tokens", 200),
            bos_token_id=lm_cfg.bos_token_id,
            eos_token_id=lm_cfg.eos_token_id,
            pad_token_id=lm_cfg.pad_token_id,
        )
        gen_kwargs.update({
            k: v for k, v in kwargs.items() if k not in _NON_GENERATE_KEYS
        })
        return self.language_model.generate(**gen_kwargs)

    # ------------------------------------------------------------------
    # Helpers for PEFT / freezing
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None):
        return self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of=pad_to_multiple_of
        )
