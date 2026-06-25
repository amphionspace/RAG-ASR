# evaluation

数据集级离线评测与压测：精度（CER / 一致性）和时延对比。脚本面向整份数据集批量运行，公共逻辑在 `src/rag_asr/`，运行前需要对应在线服务已启动。产物默认写到 `var/`，不随仓库分发。

与 `tests/` 的区别：`tests/` 只放无网络、无本机服务依赖的 pytest 单测；本目录脚本依赖在线 Triton / vLLM，不应被 pytest 收集。

## 评测脚本

| 脚本 | 语义 |
|------|------|
| `benchmark_vllm_encoder_bypass.py` | 批量对比纯 vLLM encoder 与 Triton `audio_embeds` bypass 的 CER 与时延 |
| `benchmark_triton_vs_local.py` | v1 Triton 与本地 Python 的一致性和时延对比 |
| `benchmark_triton_v2_batch.py` | v2 batch 协议压测 |

## 运行前置

- 先启动 Triton 在线服务：`bash scripts/start_triton.sh`；端口取自 `configs/serve.yaml` 的 `triton.http_port`。
- `benchmark_vllm_encoder_bypass.py` 还需要 vLLM 以 `--enable-mm-embeds` 启动。
- 依赖：`pip install -e ".[eval]"`（requests、librosa）；Triton 客户端 `pip install -e ".[triton]"`。

## 示例命令

```bash
python evaluation/benchmark_vllm_encoder_bypass.py \
  --dataset /home/ubuntu/data/testdata/base_v2_kespeech_gpu1 \
  --vllm-url http://localhost:8009 \
  --triton-url localhost:10001 \
  --limit 50
```

逐条结果写到 `--output`（默认 `var/benchmarks/vllm_encoder_bypass/*.jsonl`），汇总写到 `--summary-output`。
