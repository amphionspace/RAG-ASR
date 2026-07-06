"""Triton Python backend for RAG-ASR hotword retrieval."""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

CONTROL_TRAILER_MARKER = np.float32(16777216.0)
CONTROL_REQUEST_ID_PREFIX = "ctl:"


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


def _decode_control_array(name: str, array: np.ndarray) -> dict[str, object]:
    if array.size == 0:
        return {}
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype,
        np.floating,
    ):
        raw = bytes(array.astype(np.uint8).reshape(-1).tolist())
    else:
        value = array.reshape(-1)[0]
        if isinstance(value, bytes):
            raw = value
        else:
            raw = str(value).encode("utf-8")
    if not raw:
        return {}
    text = raw.decode("utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"invalid {name} dtype={array.dtype} shape={array.shape} "
                f"text={text!r}"
            ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return data


def _optional_control_tensor(request, name: str) -> dict[str, object]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return {}
    return _decode_control_array(name, tensor.as_numpy())


def _optional_request_id_control(request) -> dict[str, object]:
    get_request_id = getattr(request, "request_id", None)
    if get_request_id is None:
        return {}
    raw = get_request_id()
    if raw is None:
        return {}
    value = _decode_string(raw)
    if not value.startswith(CONTROL_REQUEST_ID_PREFIX):
        return {}
    encoded = value[len(CONTROL_REQUEST_ID_PREFIX) :]
    padding = "=" * (-len(encoded) % 4)
    data = json.loads(base64.urlsafe_b64decode(encoded + padding).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request_id control payload must decode to a JSON object")
    return data


def _optional_control(request) -> dict[str, object]:
    return _optional_request_id_control(request) or _optional_control_tensor(
        request,
        "CONTROL_JSON",
    )


def _split_control_trailer(wav: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if audio.size < 2 or float(audio[-1]) != float(CONTROL_TRAILER_MARKER):
        return audio, {}
    length = int(round(float(audio[-2])))
    if length < 0 or length > audio.size - 2:
        return audio, {}
    start = audio.size - 2 - length
    raw = bytes(audio[start : audio.size - 2].astype(np.uint8).tolist())
    if not raw:
        return audio[:start], {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("WAV control trailer must decode to a JSON object")
    return audio[:start], data


def _control_string(control: dict[str, object], name: str) -> str | None:
    value = control.get(name)
    if value is None:
        return None
    return str(value)


def _parse_hotwords(request, control: dict[str, object] | None = None) -> list[object]:
    if control and "hotwords" in control:
        data = control.get("hotwords")
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            return [data]
        raise ValueError("control hotwords must be a JSON string or JSON array")
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


def _requested_output_names(request) -> set[str] | None:
    for attr in ("requested_output_names", "requested_outputs"):
        get_names = getattr(request, attr, None)
        if get_names is None:
            continue
        names = get_names()
        if names is None:
            return None
        return {str(name) for name in names}
    return None


def _audio_embeds_b64(frames: np.ndarray) -> str:
    """Serialize projector frames using vLLM's audio_embeds wire format."""
    import base64
    import io

    import torch

    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    with io.BytesIO() as buf:
        torch.save(tensor, buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _management_response(
    summary: dict[str, object],
    requested_outputs: set[str] | None = None,
) -> pb_utils.InferenceResponse:
    hotwords = summary.get("hotwords", [])
    meta = {k: v for k, v in summary.items() if k != "hotwords"}
    outputs = {
        "STATUS": _string_tensor("STATUS", str(summary.get("status", "ok"))),
        "MESSAGE": _string_tensor("MESSAGE", json.dumps(meta, ensure_ascii=False)),
        "HOTWORD_COUNT": pb_utils.Tensor(
            "HOTWORD_COUNT",
            np.array([int(summary.get("total_count", 0))], dtype=np.int32),
        ),
        "HOTWORD_LIST": _string_tensor(
            "HOTWORD_LIST",
            json.dumps(hotwords, ensure_ascii=False),
        ),
    }
    if requested_outputs is not None:
        outputs = {k: v for k, v in outputs.items() if k in requested_outputs}
    return pb_utils.InferenceResponse(output_tensors=list(outputs.values()))


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
        enrollment_scope_ids = []
        enrollment_ids = []
        enable_hotword_recalls = []
        want_target_audio_embeds = []
        want_enrollment_audio_embeds = []
        request_ids = []
        infer_indices = []

        for i, request in enumerate(requests):
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
            wav = None
            wav_control: dict[str, object] = {}
            if wav_tensor is not None:
                wav, wav_control = _split_control_trailer(
                    wav_tensor.as_numpy().astype(np.float32),
                )
            control = wav_control or _optional_control(request)
            requested_outputs = _requested_output_names(request)
            action = (
                _control_string(control, "action")
                or _optional_string(request, "ACTION")
                or "infer"
            ).strip().lower()
            hotword_pool_id = (
                _control_string(control, "hotword_pool_id")
            )
            enrollment_id = _control_string(control, "enrollment_id")
            enrollment_scope_id = _control_string(control, "enrollment_scope_id")
            try:
                if action in {"infer", "retrieve"}:
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                        if _optional_bool(request, "WANT_ENROLLMENT_AUDIO_EMBEDS", False):
                            enrollment_id = _optional_string(request, "ENROLLMENT_ID")
                            enrollment_scope_id = _optional_string(
                                request,
                                "ENROLLMENT_SCOPE_ID",
                            )
                    if wav is None:
                        responses[i] = _error_response("WAV is required for infer")
                        continue
                    wavs.append(wav)
                    sample_rates.append(_optional_int(request, "SAMPLE_RATE", 16000))
                    top_ks.append(_optional_int(request, "TOP_K"))
                    hotword_pool_ids.append(hotword_pool_id)
                    enrollment_scope_ids.append(enrollment_scope_id)
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
                    request_ids.append(_control_string(control, "request_id"))
                    infer_indices.append(i)
                elif action == "upsert_enrollment":
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                        enrollment_id = _optional_string(request, "ENROLLMENT_ID")
                        enrollment_scope_id = _optional_string(
                            request,
                            "ENROLLMENT_SCOPE_ID",
                        )
                    if not enrollment_id:
                        responses[i] = _error_response(
                            "ENROLLMENT_ID is required for upsert_enrollment"
                        )
                        continue
                    if wav is None:
                        responses[i] = _error_response(
                            "WAV is required for upsert_enrollment"
                        )
                        continue
                    summary = self.retriever.upsert_enrollment(
                        wav,
                        sample_rate=_optional_int(request, "SAMPLE_RATE", 16000),
                        enrollment_id=enrollment_id,
                        enrollment_scope_id=enrollment_scope_id or hotword_pool_id,
                    )
                    responses[i] = _management_response(summary, requested_outputs)
                elif action == "get_enrollment":
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                        enrollment_id = _optional_string(request, "ENROLLMENT_ID")
                        enrollment_scope_id = _optional_string(
                            request,
                            "ENROLLMENT_SCOPE_ID",
                        )
                    if not enrollment_id:
                        responses[i] = _error_response(
                            "ENROLLMENT_ID is required for get_enrollment"
                        )
                        continue
                    summary = self.retriever.get_enrollment(
                        enrollment_id=enrollment_id,
                        enrollment_scope_id=enrollment_scope_id or hotword_pool_id,
                    )
                    responses[i] = _management_response(summary, requested_outputs)
                elif action == "delete_enrollment":
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                        enrollment_id = _optional_string(request, "ENROLLMENT_ID")
                        enrollment_scope_id = _optional_string(
                            request,
                            "ENROLLMENT_SCOPE_ID",
                        )
                    if not enrollment_id:
                        responses[i] = _error_response(
                            "ENROLLMENT_ID is required for delete_enrollment"
                        )
                        continue
                    summary = self.retriever.delete_enrollment(
                        enrollment_id=enrollment_id,
                        enrollment_scope_id=enrollment_scope_id or hotword_pool_id,
                    )
                    responses[i] = _management_response(summary, requested_outputs)
                elif action == "list":
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                    summary = self.retriever.list_hotwords(
                        user_id=hotword_pool_id,
                        query=_control_string(control, "query") if control else None,
                        limit=_optional_int(request, "LIMIT"),
                        offset=_optional_int(request, "OFFSET", 0) or 0,
                    )
                    responses[i] = _management_response(summary, requested_outputs)
                elif action == "add":
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                    summary = self.retriever.add_hotwords(
                        _parse_hotwords(request, control),
                        user_id=hotword_pool_id,
                    )
                    responses[i] = _management_response(summary, requested_outputs)
                elif action in {"delete", "remove"}:
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                    summary = self.retriever.delete_hotwords(
                        _parse_hotwords(request, control),
                        user_id=hotword_pool_id,
                    )
                    responses[i] = _management_response(summary, requested_outputs)
                elif action == "reload":
                    if not control:
                        hotword_pool_id = _optional_string(request, "HOTWORD_POOL_ID")
                    summary = self.retriever.reload_hotwords(user_id=hotword_pool_id)
                    responses[i] = _management_response(summary, requested_outputs)
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
                    enrollment_scope_ids=enrollment_scope_ids,
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
