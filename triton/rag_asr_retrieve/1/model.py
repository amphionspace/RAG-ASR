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


def _optional_bool(request, name: str, default: bool = False) -> bool:
    value = _optional_int(request, name, 1 if default else 0)
    return bool(value)


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


def _infer_message(result, *, request_id: str | None = None) -> str:
    payload: dict[str, object] = {
        "status": "ok",
        "word_count": len(result.word_list),
        "projector_len": int(result.projector_len),
    }
    if request_id:
        payload["request_id"] = request_id
    if result.enrollment_id:
        payload.update(
            {
                "enrollment_id": result.enrollment_id,
                "enrollment_projector_len": int(result.enrollment_projector_len),
                "enrollment_cache_hit": bool(result.enrollment_cache_hit),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


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
        hotword_pool_ids = []
        enrollment_user_ids = []
        enrollment_ids = []
        enable_hotword_recalls = []
        want_target_audio_embeds = []
        want_enrollment_audio_embeds = []
        request_ids = []
        infer_indices = []

        for i, request in enumerate(requests):
            action = (_optional_string(request, "ACTION") or "infer").strip().lower()
            legacy_user_id = _optional_string(request, "USER_ID")
            hotword_pool_id = (
                _optional_string(request, "HOTWORD_POOL_ID") or legacy_user_id
            )
            enrollment_id = _optional_string(request, "ENROLLMENT_ID")
            enrollment_user_id = _optional_string(request, "ENROLLMENT_USER_ID")
            enrollment_owner_id = enrollment_user_id or hotword_pool_id
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
                    hotword_pool_ids.append(hotword_pool_id)
                    enrollment_user_ids.append(enrollment_user_id)
                    enrollment_ids.append(enrollment_id)
                    enable_hotword_recalls.append(
                        _optional_bool(request, "ENABLE_HOTWORD_RECALL", True)
                    )
                    want_target_audio_embeds.append(
                        _optional_bool(request, "WANT_TARGET_AUDIO_EMBEDS", True)
                    )
                    want_enrollment_audio_embeds.append(
                        _optional_bool(
                            request,
                            "WANT_ENROLLMENT_AUDIO_EMBEDS",
                            bool(enrollment_id),
                        )
                    )
                    request_ids.append(_optional_string(request, "REQUEST_ID"))
                    infer_indices.append(i)
                elif action == "upsert_enrollment":
                    if not enrollment_id:
                        responses[i] = _error_response(
                            "ENROLLMENT_ID is required for upsert_enrollment"
                        )
                        continue
                    wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
                    if wav_tensor is None:
                        responses[i] = _error_response(
                            "WAV is required for upsert_enrollment"
                        )
                        continue
                    summary = self.retriever.upsert_enrollment(
                        wav_tensor.as_numpy().astype(np.float32),
                        sample_rate=_optional_int(request, "SAMPLE_RATE", 16000),
                        enrollment_id=enrollment_id,
                        user_id=enrollment_owner_id,
                    )
                    responses[i] = _management_response(summary)
                elif action == "get_enrollment":
                    if not enrollment_id:
                        responses[i] = _error_response(
                            "ENROLLMENT_ID is required for get_enrollment"
                        )
                        continue
                    summary = self.retriever.get_enrollment(
                        enrollment_id=enrollment_id,
                        user_id=enrollment_owner_id,
                    )
                    responses[i] = _management_response(summary)
                elif action == "delete_enrollment":
                    if not enrollment_id:
                        responses[i] = _error_response(
                            "ENROLLMENT_ID is required for delete_enrollment"
                        )
                        continue
                    summary = self.retriever.delete_enrollment(
                        enrollment_id=enrollment_id,
                        user_id=enrollment_owner_id,
                    )
                    responses[i] = _management_response(summary)
                elif action == "list":
                    summary = self.retriever.list_hotwords(
                        user_id=hotword_pool_id,
                        query=_optional_string(request, "QUERY"),
                        limit=_optional_int(request, "LIMIT"),
                        offset=_optional_int(request, "OFFSET", 0) or 0,
                    )
                    responses[i] = _management_response(summary)
                elif action == "add":
                    summary = self.retriever.add_hotwords(
                        _parse_hotwords(request),
                        user_id=hotword_pool_id,
                    )
                    responses[i] = _management_response(summary)
                elif action in {"delete", "remove"}:
                    summary = self.retriever.delete_hotwords(
                        _parse_hotwords(request),
                        user_id=hotword_pool_id,
                    )
                    responses[i] = _management_response(summary)
                elif action == "reload":
                    summary = self.retriever.reload_hotwords(user_id=hotword_pool_id)
                    responses[i] = _management_response(summary)
                else:
                    responses[i] = _error_response(f"unknown ACTION: {action}")
            except Exception as exc:
                responses[i] = _error_response(str(exc))

        if wavs:
            try:
                results = self.retriever.infer_many(
                    wavs,
                    sample_rates=sample_rates,
                    top_ks=top_ks,
                    user_ids=hotword_pool_ids,
                    enable_hotword_recalls=enable_hotword_recalls,
                    enrollment_ids=enrollment_ids,
                    enrollment_user_ids=enrollment_user_ids,
                )
                for offset, (request_index, result) in enumerate(
                    zip(infer_indices, results)
                ):
                    responses[request_index] = self._infer_response(
                        result,
                        want_target_audio_embeds=want_target_audio_embeds[offset],
                        want_enrollment_audio_embeds=want_enrollment_audio_embeds[
                            offset
                        ],
                        request_id=request_ids[offset],
                    )
            except Exception as exc:
                for request_index in infer_indices:
                    responses[request_index] = _error_response(str(exc))

        return [
            response if response is not None else _error_response("internal response missing")
            for response in responses
        ]

    @staticmethod
    def _infer_response(
        result,
        *,
        want_target_audio_embeds: bool = True,
        want_enrollment_audio_embeds: bool = False,
        request_id: str | None = None,
    ) -> pb_utils.InferenceResponse:
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
        out_audio_embeds_b64 = (
            _audio_embeds_b64(result.projector_out[: result.projector_len])
            if want_target_audio_embeds
            else ""
        )
        out_audio_embeds = pb_utils.Tensor(
            "AUDIO_EMBEDS_B64",
            np.array(
                [out_audio_embeds_b64],
                dtype=object,
            ),
        )
        enrollment_embeds_b64 = ""
        if (
            want_enrollment_audio_embeds
            and result.enrollment_projector_out is not None
        ):
            enrollment_embeds_b64 = _audio_embeds_b64(
                result.enrollment_projector_out[
                    : result.enrollment_projector_len
                ]
            )
        out_enrollment_audio_embeds = pb_utils.Tensor(
            "ENROLLMENT_AUDIO_EMBEDS_B64",
            np.array([enrollment_embeds_b64], dtype=object),
        )
        out_enrollment_len = pb_utils.Tensor(
            "ENROLLMENT_PROJECTOR_LEN",
            np.array([int(result.enrollment_projector_len)], dtype=np.int32),
        )
        out_status = _string_tensor("STATUS", "ok")
        out_message = _string_tensor("MESSAGE", _infer_message(result, request_id=request_id))
        return pb_utils.InferenceResponse(
            output_tensors=[
                out_proj,
                out_len,
                out_words,
                out_audio_embeds,
                out_enrollment_audio_embeds,
                out_enrollment_len,
                out_status,
                out_message,
            ]
        )
