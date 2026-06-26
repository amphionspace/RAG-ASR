"""Triton Python backend for explicit-batch RAG-ASR retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))


def _optional_vector(request, name: str) -> list[int] | None:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    return [int(x) for x in tensor.as_numpy().reshape(-1).tolist()]


def _audio_embeds_b64(frames: np.ndarray) -> str:
    """Serialize projector frames using vLLM's audio_embeds wire format."""
    import base64
    import io

    import torch

    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    with io.BytesIO() as buf:
        torch.save(tensor, buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


class TritonPythonModel:
    def initialize(self, args):
        from rag_asr.serve import RAGASRRetriever

        model_config = json.loads(args["model_config"])
        params = {
            k: v["string_value"]
            for k, v in model_config.get("parameters", {}).items()
            if k != "EXECUTION_ENV_PATH"
        }
        self.packed_audio = params.pop("packed_audio", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.retriever = RAGASRRetriever.from_parameters(params)

    def execute(self, requests):
        responses = []
        for request in requests:
            wav_batch = pb_utils.get_input_tensor_by_name(
                request, "WAV_BATCH"
            ).as_numpy().astype(np.float32)
            wav_lens = pb_utils.get_input_tensor_by_name(
                request, "WAV_LEN"
            ).as_numpy().astype(np.int32)
            sample_rates = _optional_vector(request, "SAMPLE_RATE")
            top_ks = _optional_vector(request, "TOP_K")

            results = self.retriever.infer_padded_batch(
                wav_batch,
                wav_lens,
                sample_rates=sample_rates,
                top_ks=top_ks,
                packed_audio=self.packed_audio,
            )

            batch_size = len(results)
            max_projector_len = max(result.projector_len for result in results)
            projector_dim = int(results[0].projector_out.shape[1])
            projector_out = np.zeros(
                (batch_size, max_projector_len, projector_dim),
                dtype=np.float32,
            )
            projector_len = np.zeros((batch_size,), dtype=np.int32)
            word_list = np.empty((batch_size,), dtype=object)
            audio_embeds = np.empty((batch_size,), dtype=object)
            for i, result in enumerate(results):
                projector_out[i, : result.projector_len, :] = result.projector_out
                projector_len[i] = result.projector_len
                word_list[i] = json.dumps(result.word_list, ensure_ascii=False)
                audio_embeds[i] = _audio_embeds_b64(
                    result.projector_out[: result.projector_len]
                )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("PROJECTOR_OUT", projector_out),
                        pb_utils.Tensor("PROJECTOR_LEN", projector_len),
                        pb_utils.Tensor("WORD_LIST", word_list),
                        pb_utils.Tensor("AUDIO_EMBEDS_B64", audio_embeds),
                    ]
                )
            )
        return responses
