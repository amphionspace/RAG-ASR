"""Render Triton model repository config from RAG-ASR YAML config."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rag_asr.config import PROJECT_ROOT, load_config


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parameters(params: dict[str, str], *, packed_audio: str | None = None) -> str:
    items = dict(params)
    if packed_audio is not None:
        items["packed_audio"] = packed_audio
    chunks = []
    for key, value in items.items():
        chunks.append(
            "parameters: {\n"
            f'  key: "{key}"\n'
            f'  value: {{ string_value: "{_quote(value)}" }}\n'
            "}"
        )
    return "\n".join(chunks)


def _rag_asr_retrieve_config(params: dict[str, str]) -> str:
    return f'''name: "rag_asr_retrieve"
backend: "python"
max_batch_size: 0

input [
  {{
    name: "WAV"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "SAMPLE_RATE"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "TOP_K"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "ACTION"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "HOTWORDS"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "QUERY"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "LIMIT"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "OFFSET"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }}
]

output [
  {{
    name: "PROJECTOR_OUT"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }},
  {{
    name: "PROJECTOR_LEN"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "WORD_LIST"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "AUDIO_EMBEDS_B64"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "STATUS"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "MESSAGE"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "HOTWORD_COUNT"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "HOTWORD_LIST"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    # KIND_CPU: Triton CUDA runtime may not match the host driver.
    # PyTorch inside the execution env still uses GPU via its bundled CUDA.
    kind: KIND_CPU
    count: 1
  }}
]

{_parameters(params)}
'''


def _rag_asr_retrieve_v2_config(params: dict[str, str]) -> str:
    return f'''name: "rag_asr_retrieve_v2"
backend: "python"
max_batch_size: 0

input [
  {{
    name: "WAV_BATCH"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }},
  {{
    name: "WAV_LEN"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "SAMPLE_RATE"
    data_type: TYPE_INT32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "TOP_K"
    data_type: TYPE_INT32
    dims: [ -1 ]
    optional: true
  }}
]

output [
  {{
    name: "PROJECTOR_OUT"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
  }},
  {{
    name: "PROJECTOR_LEN"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "WORD_LIST"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }},
  {{
    name: "AUDIO_EMBEDS_B64"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

instance_group [
  {{
    # KIND_CPU: Triton CUDA runtime may not match the host driver.
    # PyTorch inside the execution env still uses GPU via its bundled CUDA.
    kind: KIND_CPU
    count: 1
  }}
]

{_parameters(params, packed_audio="false")}
'''


def render_model_repo(config_path: str | Path | None, output: str | Path | None) -> Path:
    cfg = load_config(config_path)
    source_repo = Path(cfg.triton.model_repo)
    if not source_repo.is_absolute():
        source_repo = PROJECT_ROOT / source_repo
    output_repo = Path(output or cfg.triton.rendered_model_repo)
    if not output_repo.is_absolute():
        output_repo = PROJECT_ROOT / output_repo

    if output_repo.exists():
        shutil.rmtree(output_repo)
    shutil.copytree(source_repo, output_repo)

    params = cfg.to_triton_parameters()
    (output_repo / "rag_asr_retrieve" / "config.pbtxt").write_text(
        _rag_asr_retrieve_config(params),
        encoding="utf-8",
    )
    v2_dir = output_repo / "rag_asr_retrieve_v2"
    if v2_dir.is_dir():
        (v2_dir / "config.pbtxt").write_text(
            _rag_asr_retrieve_v2_config(params),
            encoding="utf-8",
        )
    return output_repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="render a Triton model repository")
    render.add_argument("--config", default=None)
    render.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.command == "render":
        print(render_model_repo(args.config, args.output))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
