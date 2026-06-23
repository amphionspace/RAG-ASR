"""AmphionASR model configuration.

Defines the configuration hierarchy for the Amphion ASR model which
combines an audio encoder (Qwen3-style or Zipformer), a multi-modal
projector, and a causal language model.
"""

from transformers import PretrainedConfig, AutoConfig


class AmphionASRAudioEncoderConfig(PretrainedConfig):
    """Configuration for the Qwen3-style audio encoder tower."""

    model_type = "amphion_asr_audio_encoder"

    def __init__(
        self,
        d_model: int = 1280,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layers: int = 24,
        num_mel_bins: int = 128,
        max_source_positions: int = 6000,
        activation_function: str = "gelu",
        encoder_layerdrop: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        output_dim: int = 2048,
        conv_channels: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.activation_function = activation_function
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        self.output_dim = output_dim
        self.conv_channels = conv_channels or [256, 256, 256]


class ZipformerAudioEncoderConfig(PretrainedConfig):
    """Configuration for the Zipformer audio encoder.

    Supports ``custom`` (causal) and ``custom_noncausal`` presets.
    Default values correspond to ``custom_noncausal``.
    """

    model_type = "zipformer_audio_encoder"

    def __init__(
        self,
        feature_dim: int = 80,
        num_encoder_layers: str = "2,2,4,5,4,2",
        downsampling_factor: str = "1,2,4,8,4,2",
        feedforward_dim: str = "768,1024,1536,2048,1536,1024",
        num_heads: str = "8,8,8,12,8,8",
        encoder_dim: str = "256,384,512,768,512,384",
        query_head_dim: str = "32",
        value_head_dim: str = "12",
        pos_head_dim: str = "4",
        pos_dim: int = 48,
        encoder_unmasked_dim: str = "192,256,320,512,320,256",
        cnn_module_kernel: str = "31,31,15,15,15,31",
        causal: bool = False,
        chunk_size: str = "-1",
        left_context_frames: str = "-1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_encoder_layers = num_encoder_layers
        self.downsampling_factor = downsampling_factor
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.encoder_dim = encoder_dim
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim
        self.pos_head_dim = pos_head_dim
        self.pos_dim = pos_dim
        self.encoder_unmasked_dim = encoder_unmasked_dim
        self.cnn_module_kernel = cnn_module_kernel
        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames


class AmphionASRProjectorConfig(PretrainedConfig):
    """Configuration for the multi-modal projector between encoder and LLM."""

    model_type = "amphion_asr_projector"

    def __init__(
        self,
        encoder_dim: int = 1280,
        llm_dim: int = 2048,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.dropout = dropout


class AmphionASRConfig(PretrainedConfig):
    """
    Configuration for the full AmphionASR model.

    Composes an audio encoder config, a projector config, and a text (LLM)
    config.  When saved, the ``auto_map`` field allows ``AutoModel`` to
    locate the custom modelling code via ``trust_remote_code=True``.
    """

    model_type = "amphion_asr"

    _FEAT_TYPE_MAP = {
        "qwen3asr": "whisper",
        "qwen3omni_captioner": "whisper",
        "qwen3omni": "whisper",
        "zipformer": "kaldi_fbank",
    }

    def __init__(
        self,
        audio_encoder_config: dict = None,
        projector_config: dict = None,
        text_config: dict = None,
        num_prompt_tokens: int = 4,
        default_speech_token_id: int = None,
        start_text_token_id: int = None,
        end_text_token_id: int = None,
        start_speech_token_id: int = None,
        end_speech_token_id: int = None,
        encoder_type: str = "qwen3asr",
        feature_extractor_type: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if audio_encoder_config is None:
            audio_encoder_config = {}
        if isinstance(audio_encoder_config, dict):
            if encoder_type == "zipformer":
                audio_encoder_config = ZipformerAudioEncoderConfig(**audio_encoder_config)
            else:
                audio_encoder_config = AmphionASRAudioEncoderConfig(**audio_encoder_config)
        self.audio_encoder_config = audio_encoder_config

        if projector_config is None:
            projector_config = {}
        if isinstance(projector_config, dict):
            projector_config = AmphionASRProjectorConfig(**projector_config)
        self.projector_config = projector_config

        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            self.text_config = AutoConfig.for_model(**text_config) if "model_type" in text_config else text_config
        else:
            self.text_config = text_config

        self.num_prompt_tokens = num_prompt_tokens
        self.default_speech_token_id = default_speech_token_id
        self.start_text_token_id = start_text_token_id
        self.end_text_token_id = end_text_token_id
        self.start_speech_token_id = start_speech_token_id
        self.end_speech_token_id = end_speech_token_id
        self.encoder_type = encoder_type
        self.feature_extractor_type = (
            feature_extractor_type
            or self._FEAT_TYPE_MAP.get(encoder_type, "whisper")
        )

        self.architectures = ["AmphionASRForConditionalGeneration"]
        self.auto_map = {
            "AutoConfig": "rag_asr.backends.amphion.configuration_amphion_asr.AmphionASRConfig",
            "AutoModelForCausalLM": "rag_asr.backends.amphion.modeling_amphion_asr.AmphionASRForConditionalGeneration",
            "AutoModel": "rag_asr.backends.amphion.modeling_amphion_asr.AmphionASRForConditionalGeneration",
        }

    def to_dict(self):
        output = super().to_dict()
        if hasattr(self.audio_encoder_config, "to_dict"):
            output["audio_encoder_config"] = self.audio_encoder_config.to_dict()
        if hasattr(self.projector_config, "to_dict"):
            output["projector_config"] = self.projector_config.to_dict()
        if hasattr(self.text_config, "to_dict"):
            output["text_config"] = self.text_config.to_dict()
        return output
