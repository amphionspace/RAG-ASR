"""Triton Python backend for RAG-ASR hotword retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))


def _decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional_string(request, name: str) -> str | None:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    values = tensor.as_numpy().reshape(-1)
    if values.size == 0:
        return None
    return _decode_string(values[0])


def _optional_int(request, name: str, default: int | None = None) -> int | None:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return default
    values = tensor.as_numpy().reshape(-1)
    if values.size == 0:
        return default
    return int(values[0])


def _parse_hotwords(request) -> list[object]:
    raw = _optional_string(request, "HOTWORDS")
    if raw is None or not raw.strip():
        return []
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        return [data]
    raise ValueError("HOTWORDS must be a JSON string or JSON array")


def _string_tensor(name: str, value: str) -> pb_utils.Tensor:
    return pb_utils.Tensor(name, np.array([value], dtype=object))


def _audio_embeds_b64(frames: np.ndarray) -> str:
    """Serialize projector frames using vLLM's audio_embeds wire format."""
    import base64
    import io

    import torch

    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    with io.BytesIO() as buf:
        torch.save(tensor, buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _management_response(summary: dict[str, object]) -> pb_utils.InferenceResponse:
    hotwords = summary.get("hotwords", [])
    meta = {k: v for k, v in summary.items() if k != "hotwords"}
    return pb_utils.InferenceResponse(
        output_tensors=[
            _string_tensor("STATUS", str(summary.get("status", "ok"))),
            _string_tensor("MESSAGE", json.dumps(meta, ensure_ascii=False)),
            pb_utils.Tensor(
                "HOTWORD_COUNT",
                np.array([int(summary.get("total_count", 0))], dtype=np.int32),
            ),
            _string_tensor("HOTWORD_LIST", json.dumps(hotwords, ensure_ascii=False)),
        ]
    )


def _error_response(message: str) -> pb_utils.InferenceResponse:
    return pb_utils.InferenceResponse(error=pb_utils.TritonError(message))


class TritonPythonModel:
    def initialize(self, args):
        from rag_asr.serve import RAGASRRetriever

        model_config = json.loads(args["model_config"])
        params = {
            k: v["string_value"]
            for k, v in model_config.get("parameters", {}).items()
            if k != "EXECUTION_ENV_PATH"
        }
        self.retriever = RAGASRRetriever.from_parameters(params)

    def execute(self, requests):
        responses: list[pb_utils.InferenceResponse | None] = [None] * len(requests)
        wavs = []
        sample_rates = []
        top_ks = []
        infer_indices = []

        for i, request in enumerate(requests):
            action = (_optional_string(request, "ACTION") or "infer").strip().lower()
            try:
                if action in {"infer", "retrieve"}:
                    wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
                    if wav_tensor is None:
                        responses[i] = _error_response("WAV is required for infer")
                        continue
                    wav = wav_tensor.as_numpy().astype(np.float32)
                    wavs.append(wav)
                    sample_rates.append(_optional_int(request, "SAMPLE_RATE", 16000))
                    top_ks.append(_optional_int(request, "TOP_K"))
                    infer_indices.append(i)
                elif action == "list":
                    summary = self.retriever.list_hotwords(
                        query=_optional_string(request, "QUERY"),
                        limit=_optional_int(request, "LIMIT"),
                        offset=_optional_int(request, "OFFSET", 0) or 0,
                    )
                    responses[i] = _management_response(summary)
                elif action == "add":
                    summary = self.retriever.add_hotwords(_parse_hotwords(request))
                    responses[i] = _management_response(summary)
                elif action in {"delete", "remove"}:
                    summary = self.retriever.delete_hotwords(_parse_hotwords(request))
                    responses[i] = _management_response(summary)
                elif action == "reload":
                    summary = self.retriever.reload_hotwords()
                    responses[i] = _management_response(summary)
                else:
                    responses[i] = _error_response(f"unknown ACTION: {action}")
            except Exception as exc:
                responses[i] = _error_response(str(exc))

        if wavs:
            results = self.retriever.infer_many(
                wavs,
                sample_rates=sample_rates,
                top_ks=top_ks,
            )
            for request_index, result in zip(infer_indices, results):
                responses[request_index] = self._infer_response(result)

        return [
            response if response is not None else _error_response("internal response missing")
            for response in responses
        ]

    @staticmethod
    def _infer_response(result) -> pb_utils.InferenceResponse:
        out_proj = pb_utils.Tensor(
            "PROJECTOR_OUT",
            result.projector_out,
        )
        out_len = pb_utils.Tensor(
            "PROJECTOR_LEN",
            np.array([result.projector_len], dtype=np.int32),
        )
        out_words = pb_utils.Tensor(
            "WORD_LIST",
            np.array([json.dumps(result.word_list, ensure_ascii=False)], dtype=object),
        )
        out_audio_embeds = pb_utils.Tensor(
            "AUDIO_EMBEDS_B64",
            np.array(
                [_audio_embeds_b64(result.projector_out[: result.projector_len])],
                dtype=object,
            ),
        )
        return pb_utils.InferenceResponse(
            output_tensors=[out_proj, out_len, out_words, out_audio_embeds]
        )
