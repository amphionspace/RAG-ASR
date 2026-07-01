# examples

可独立运行的最小示例与冒烟脚本，以及配套的示例数据。脚本只做演示和端到端连通性验证，公共逻辑在 `src/rag_asr/`，运行前需要对应在线服务已启动。

## 示例脚本

| 脚本 | 语义 |
|------|------|
| `triton_client_example.py` | v1 Triton `rag_asr_retrieve` 单条音频最小调用 |
| `triton_v2_batch_example.py` | v2 显式 batch 协议（`rag_asr_retrieve_v2`）调用示例 |
| `triton_recall_check.py` | 对本目录样例做 Triton 端到端 recall 冒烟，输出召回命中与 PRRR |
| `vllm_encoder_bypass.py` | 单条音频对比 vLLM 原始 encoder 与 Triton `PROJECTOR_OUT` → vLLM `audio_embeds` bypass |

## 示例数据

`metadata.jsonl`、`transcripts.tsv`、`hotwords.tsv`、`hotword_pool.txt`、`wav.scp` 是冒烟脚本使用的小样例；`hotword_pool.txt` 也是 `configs/serve.yaml` 默认使用的只读种子词池。在线服务管理热词时会写入 `var/hotwords/<user>.txt`，不会修改本目录样例。

## 运行前置

- 先启动 Triton 在线服务：`bash scripts/start_triton.sh`；Triton HTTP 端口取自 `configs/serve.yaml` 的 `triton.http_port`，用 `--url` / `--triton-url` 对齐。
- `vllm_encoder_bypass.py` 还需要 vLLM 以 `--enable-mm-embeds` 启动，并用 `--vllm-url` 指向该服务。
- 依赖：Triton 客户端 `pip install -e ".[triton]"`；bypass 还需 `pip install -e ".[eval]"`（requests、librosa）。

## 示例命令

```bash
# v1 单条调用
python examples/triton_client_example.py --url localhost:10001 --wav examples/audio/xxx.wav

# 样例 recall 冒烟
python examples/triton_recall_check.py --url localhost:10001

# vLLM encoder bypass（需 vLLM --enable-mm-embeds）
python examples/vllm_encoder_bypass.py \
  --vllm-url http://localhost:8009 \
  --triton-url localhost:10001 \
  --wav examples/audio/xxx.wav
```
