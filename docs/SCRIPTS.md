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

## 路径和环境变量

脚本默认只假设项目内相对路径；外部数据盘和共享输出目录需要通过环境变量传入。优先设置少量根目录变量，目录结构不一致时再用细粒度变量覆盖。

通用变量：

- `RAG_ASR_DATA_ROOT`：外部 ASR 数据根目录，用于推导 Common Voice、GigaSpeech、AISHELL 等 manifest 位置。
- `RAG_ASR_HOTWORD_ROOT`：热词词表根目录；未设置时，若有 `RAG_ASR_DATA_ROOT`，默认推导为 `${RAG_ASR_DATA_ROOT}/hotword`。
- `RAG_ASR_BASE_MODEL`：覆盖项目内默认基座模型目录 `checkpoints/base/amphion_1.7b_merged`。
- `RAG_ASR_ADAPTER`：离线推理 adapter checkpoint；默认使用 `${RAG_ASR_BASE_MODEL}/hotword_adapter/best_adapter.pt`。

`scripts/build_triton_exec_env.sh` 会按 `CONDA`、`CONDA_EXE`、`conda` 命令、`CONDA_PREFIX` 的顺序寻找 conda。输出归档默认写到 `var/triton-exec-env.tar.gz`，可用 `RAG_ASR_TRITON_EXEC_ENV_TAR` 指向共享盘；环境名可用 `RAG_ASR_TRITON_EXEC_ENV_NAME` 覆盖。Triton Python stub symlink 可用 `RAG_ASR_TRITON_PYTHON_STUB_LINK` 覆盖，设为 `off` 或 `none` 时 `start_triton.sh` 不自动创建。

`scripts/infer.sh` 使用 `RAG_ASR_DATA_ROOT` 推导中英文测试 manifest，并使用 `RAG_ASR_HOTWORD_ROOT` 推导 `zh-10k.txt` 和 `en-10k.txt`。目录不一致时可设置 `RAG_ASR_CV_ZH_INFER_DIR`、`RAG_ASR_CV_EN_INFER_DIR`、`RAG_ASR_ZH_HOTWORD_POOL`、`RAG_ASR_EN_HOTWORD_POOL`，或直接设置 `RAG_ASR_ZH_SUPERVISIONS`、`RAG_ASR_ZH_RECORDINGS`、`RAG_ASR_EN_SUPERVISIONS`、`RAG_ASR_EN_RECORDINGS`。GPU 列表来自 `RAG_ASR_INFER_GPUS`，未设置时使用 `CUDA_VISIBLE_DEVICES`，再退回单卡 `0`。

`scripts/train_retrieval.sh` 使用 `RAG_ASR_DATA_ROOT` 推导各训练语料 manifest 目录；目录不一致时可分别设置 `RAG_ASR_TRAIN_V1`、`RAG_ASR_CV_EN_HOTWORD_DIR`、`RAG_ASR_CV_ZH_HOTWORD_DIR`、`RAG_ASR_GIGASPEECH_HOTWORD_DIR`、`RAG_ASR_AISHELL_HOTWORD_DIR`、`RAG_ASR_AISHELL2_HOTWORD_DIR`、`RAG_ASR_AISHELL3_HOTWORD_DIR`、`RAG_ASR_MAGICDATA_HOTWORD_DIR`、`RAG_ASR_THCHS30_HOTWORD_DIR`、`RAG_ASR_ZHVOICE_HOTWORD_DIR`。训练卡数默认从 `CUDA_VISIBLE_DEVICES` 推导，可用 `RAG_ASR_NUM_GPUS` 覆盖。

压测脚本默认读取 `configs/serve.yaml` 或 `RAG_ASR_CONFIG`，并提供 `--config`、`--base-model-path`、`--adapter-ckpt`、`--hotword-pool-file`、`--cache-dir`、`--device` 做临时覆盖。

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
