# Scripts

`scripts/` 里不是所有文件都是日常入口。普通使用优先只看 README 中的最少入口；本页说明其余脚本的职责和迁移方向。

## 日常入口

| 脚本 | 语义 |
|------|------|
| `scripts/start_triton.sh` | 启动 Triton 在线服务；读取 `configs/serve.yaml` 或 `RAG_ASR_CONFIG`，渲染运行时 model repository 到 `var/triton_repo` |
| `scripts/hotword_status.sh` | 查看当前 Triton 热词服务 live/ready、热词总量、样例热词和关键配置；默认连 `localhost:8000` |
| `scripts/triton_hotword_client.py` | Triton 客户端；负责热词库管理和单条音频检索 |
| `scripts/infer.sh` | 离线批量检索，用于数据集评测或生成 `hw_map` |
| `scripts/train_retrieval.sh` | 训练双塔检索 adapter |
| `scripts/build_triton_exec_env.sh` | 首次部署或重建 Triton Python backend 执行环境 |

## 调试入口

| 脚本 | 语义 |
|------|------|
| `scripts/serve_http.sh` | 本地 FastAPI 调试服务；读取同一份 `configs/serve.yaml` |
| `scripts/serve_http.py` | `serve_http.sh` 调用的 Python 服务实现 |

## 离线内部依赖

| 脚本 | 语义 |
|------|------|
| `scripts/retrieve.py` | 离线检索实现；当前被 `infer.sh` 和 `rag-asr-retrieve` 间接使用 |
| `scripts/merge_hw_maps.py` | 旧式兼容入口；优先使用 `rag-asr-merge-shards` |

## 冒烟测试和示例

| 脚本 | 语义 |
|------|------|
| `scripts/triton_client_test.py` | v1 Triton 单条音频最小调用；功能已被 `triton_hotword_client.py infer` 覆盖 |
| `scripts/triton_v2_client_test.py` | v2 显式 batch 协议示例 |
| `scripts/test_triton_examples.py` | 对 `examples/` 样例做 Triton 端到端 recall 验证 |
| `scripts/test_vllm_encoder_bypass.py` | 对比 vLLM 原始 audio encoder 路径与 Triton `PROJECTOR_OUT` → vLLM `audio_embeds` bypass 路径 |

## 压测和对比

| 脚本 | 语义 |
|------|------|
| `scripts/benchmark_triton_vs_local.py` | v1 Triton 与本地 Python 的一致性和时延对比 |
| `scripts/benchmark_triton_v2_batch.py` | v2 batch 协议压测 |

## 后续整理方向

- 冒烟测试脚本可迁到 `examples/`。
- 压测脚本可迁到 `tools/benchmark/`。
- `retrieve.py` 适合逐步内收到 `src/rag_asr/cli_retrieve.py`，让 `rag-asr-retrieve` 不再依赖脚本路径。
- `merge_hw_maps.py` 等兼容入口等 `infer.sh` 完全改用 console script 后再移动或删除。
