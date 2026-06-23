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

from rag_asr.serve import RAGASRRetriever, word_list_json


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = {
            k: v["string_value"]
            for k, v in model_config.get("parameters", {}).items()
            if k != "EXECUTION_ENV_PATH"
        }
        self.retriever = RAGASRRetriever.from_parameters(params)

    def execute(self, requests):
        wavs = []
        sample_rates = []
        top_ks = []
        for request in requests:
            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy().astype(
                np.float32
            )
            sr_tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
            sample_rate = int(sr_tensor.as_numpy()[0]) if sr_tensor is not None else 16000
            top_k_tensor = pb_utils.get_input_tensor_by_name(request, "TOP_K")
            top_k = int(top_k_tensor.as_numpy()[0]) if top_k_tensor is not None else None

            wavs.append(wav)
            sample_rates.append(sample_rate)
            top_ks.append(top_k)

        results = self.retriever.infer_many(
            wavs,
            sample_rates=sample_rates,
            top_ks=top_ks,
        )

        responses = []
        for result in results:
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
                np.array([word_list_json(result.word_list)], dtype=object),
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_proj, out_len, out_words])
            )
        return responses
