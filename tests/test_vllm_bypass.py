"""Offline unit tests for the pure helpers in ``rag_asr.vllm_bypass``.

These cover only the network-free, dependency-light functions so the suite runs
without Triton, vLLM, torch or a live service.
"""

from __future__ import annotations

import pytest

from rag_asr.vllm_bypass import (
    build_messages,
    normalize_triton_url,
    normalize_vllm_url,
    stable_embed_uuid,
)


def test_normalize_vllm_url_adds_scheme_and_strips_slash():
    assert normalize_vllm_url("localhost:8009") == "http://localhost:8009"
    assert normalize_vllm_url("http://localhost:8009/") == "http://localhost:8009"
    assert normalize_vllm_url(" https://host:1/ ") == "https://host:1"


def test_normalize_triton_url_reduces_to_netloc():
    assert normalize_triton_url("localhost:8000") == "localhost:8000"
    assert normalize_triton_url("http://localhost:8000/") == "localhost:8000"
    assert normalize_triton_url("grpc://1.2.3.4:9001") == "1.2.3.4:9001"


def test_stable_embed_uuid_is_deterministic_and_prefixed():
    first = stable_embed_uuid("abc")
    again = stable_embed_uuid("abc")
    other = stable_embed_uuid("abd")

    assert first == again
    assert first != other
    assert first.startswith("triton-audio-")
    assert len(first) == len("triton-audio-") + 16


def test_build_messages_qwen3_asr_user_is_audio_only():
    block = {"type": "audio_embeds", "audio_embeds": "x", "uuid": "u"}
    messages = build_messages(
        block, prompt_style="qwen3_asr", hotwords=["北京"], language="zh-cn"
    )

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "Hotwords: 北京"
    assert messages[1]["content"] == [block]


def test_build_messages_qwen3_asr_empty_system_without_hotwords():
    block = {"type": "input_audio"}
    messages = build_messages(
        block, prompt_style="qwen3_asr", hotwords=[], language="zh-cn"
    )

    assert messages[0]["content"] == ""
    assert messages[1]["content"] == [block]


def test_build_messages_swift_puts_text_before_audio():
    block = {"type": "input_audio"}
    messages = build_messages(
        block, prompt_style="swift", hotwords=["a", "b"], language="en"
    )

    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0]["type"] == "text"
    assert "Transcribe the following audio." in content[0]["text"]
    assert "Language: en" in content[0]["text"]
    assert "Hotwords: a,b" in content[0]["text"]
    assert content[1] is block


def test_build_messages_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown prompt style"):
        build_messages({}, prompt_style="nope", hotwords=[], language="zh")
