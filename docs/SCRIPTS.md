# Scripts

`scripts/` 里不是所有文件都是日常入口。普通使用优先只看 README 中的最少入口；本页说明其余脚本的职责和迁移方向。

## 日常入口

| 脚本 | 语义 |
|------|------|
| `scripts/start_triton.sh` | 启动 Triton 在线服务；读取 `configs/serve.yaml` 或 `RAG_ASR_CONFIG`，渲染运行时 model repository 到 `var/triton_repo` |
| `scripts/hotword_status.sh` | 查看当前 Triton 热词服务 live/ready、指定用户热词总量、样例热词和关键配置；默认连 `localhost:8000`、用户为 `default` |
| `scripts/triton_hotword_client.py` | Triton 客户端；通过 `--user` 管理指定上游用户的热词库并做单条音频检索 |
| `scripts/infer.sh` | 离线批量检索，用于数据集评测或生成 `hw_map` |
| `scripts/train_retrieval.sh` | 训练双塔检索 adapter |
| `scripts/pull_triton_image.sh` | 拉取官方 Triton Docker 镜像并抽取 `/opt/tritonserver` 到本地，内置重试，适合网络不稳定时放在 `tmux` 中运行 |
| `scripts/build_envs.sh` | 首次部署或重建 Triton 在线服务所需的全部本地环境 |
| `scripts/build_triton_server_env.sh` | 首次部署或重建 Triton server 启动环境 |
| `scripts/build_triton_exec_env.sh` | 首次部署或重建 Triton Python backend 执行环境 |

## 服务配置

在线服务只维护一个配置入口：`configs/serve.yaml`。`scripts/start_triton.sh`、`scripts/serve_http.sh` 和压测脚本默认读取这份配置；如需维护另一套部署配置，可以用 `RAG_ASR_CONFIG` 指向同 schema 的 YAML 文件。

## 离线脚本环境变量

离线推理、训练和环境构建脚本默认只假设项目内相对路径；外部数据盘和共享输出目录仍通过环境变量传入。优先设置少量根目录变量，目录结构不一致时再用细粒度变量覆盖。

离线通用变量：

- `RAG_ASR_DATA_ROOT`：外部 ASR 数据根目录，用于推导 Common Voice、GigaSpeech、AISHELL 等 manifest 位置。
- `RAG_ASR_HOTWORD_ROOT`：热词词表根目录；未设置时，若有 `RAG_ASR_DATA_ROOT`，默认推导为 `${RAG_ASR_DATA_ROOT}/hotword`。
- `RAG_ASR_BASE_MODEL`：覆盖项目内默认基座模型目录 `checkpoints/base/amphion_1.7b_merged`。
- `RAG_ASR_ADAPTER`：离线推理 adapter checkpoint；默认使用 `${RAG_ASR_BASE_MODEL}/hotword_adapter/best_adapter.pt`。

`scripts/build_envs.sh` 是在线服务环境的一键总入口，按顺序调用 `build_triton_server_env.sh` 和 `build_triton_exec_env.sh`。如果只想重建其中一层，可设置 `RAG_ASR_SKIP_TRITON_SERVER_ENV=1` 或 `RAG_ASR_SKIP_TRITON_EXEC_ENV=1`。

`scripts/pull_triton_image.sh` 用于准备官方 Triton server 本体。默认拉取 `nvcr.io/nvidia/tritonserver:24.10-py3`，失败后每 60 秒重试，`RAG_ASR_DOCKER_PULL_MAX_ATTEMPTS=0` 表示无限重试。pull 成功后默认抽取镜像内 `/opt/tritonserver` 到 `var/tritonserver`，补充复制 host-mode 所需的少量镜像系统库，并用 `ldd` 检查 `tritonserver` 是否存在缺失 `.so` 或 glibc 版本不兼容。如果目标目录已经包含 `bin/tritonserver` 和 `backends/python` 则跳过抽取。常用覆盖项：`RAG_ASR_TRITON_IMAGE` 指定镜像，`RAG_ASR_TRITONSERVER_ROOT` 指定抽取目录，`RAG_ASR_TRITON_EXTRACT_FORCE=1` 强制替换，`RAG_ASR_TRITON_EXTRACT_AFTER_PULL=0` 只下载不抽取，`RAG_ASR_BUILD_TRITON_SERVER_ENV_AFTER_EXTRACT=1` 在抽取后继续调用 `build_triton_server_env.sh`。

`scripts/build_triton_server_env.sh` 创建启动 Triton server 的 conda 环境，默认名为 `triton`，安装 `tritonclient[http]` 和 `PyYAML`，并绑定已有官方 Triton server 安装。Triton server 本体不是 pip 包；裸机部署需先准备 `/opt/tritonserver` 这类官方安装，或用 `RAG_ASR_TRITONSERVER_ROOT` 指定实际路径。安装官方 Triton server 本体的步骤见 [docs/SERVICE.md](SERVICE.md)。环境名可用 `RAG_ASR_TRITON_SERVER_ENV_NAME` 覆盖；启动时 `scripts/start_triton.sh` 默认激活同名环境，也可用 `RAG_ASR_TRITON_CONDA_ENV` 指向已有 server 环境。

`scripts/build_triton_exec_env.sh` 创建 Triton Python backend 的执行环境，默认名为 `triton-exec`。它会按 `CONDA`、`CONDA_EXE`、`conda` 命令、`CONDA_PREFIX` 的顺序寻找 conda。输出归档默认写到 `var/triton-exec-env.tar.gz`，可用 `RAG_ASR_TRITON_EXEC_ENV_TAR` 指向共享盘；环境名可用 `RAG_ASR_TRITON_EXEC_ENV_NAME` 覆盖。该执行环境的 Python 版本必须匹配 Triton Python backend stub；默认 `24.10-py3` 对应 Python 3.10，可用 `RAG_ASR_TRITON_EXEC_PYTHON` 覆盖。Triton Python stub symlink 在 `configs/serve.yaml` 的 `triton.python_stub_link` 配置，设为 `off` 或 `none` 时 `start_triton.sh` 不自动创建。

`scripts/infer.sh` 使用 `RAG_ASR_DATA_ROOT` 推导中英文测试 manifest，并使用 `RAG_ASR_HOTWORD_ROOT` 推导 `zh-10k.txt` 和 `en-10k.txt`。目录不一致时可设置 `RAG_ASR_CV_ZH_INFER_DIR`、`RAG_ASR_CV_EN_INFER_DIR`、`RAG_ASR_ZH_HOTWORD_POOL`、`RAG_ASR_EN_HOTWORD_POOL`，或直接设置 `RAG_ASR_ZH_SUPERVISIONS`、`RAG_ASR_ZH_RECORDINGS`、`RAG_ASR_EN_SUPERVISIONS`、`RAG_ASR_EN_RECORDINGS`。GPU 列表来自 `RAG_ASR_INFER_GPUS`，未设置时使用 `CUDA_VISIBLE_DEVICES`，再退回单卡 `0`。

`scripts/train_retrieval.sh` 使用 `RAG_ASR_DATA_ROOT` 推导各训练语料 manifest 目录；目录不一致时可分别设置 `RAG_ASR_TRAIN_V1`、`RAG_ASR_CV_EN_HOTWORD_DIR`、`RAG_ASR_CV_ZH_HOTWORD_DIR`、`RAG_ASR_GIGASPEECH_HOTWORD_DIR`、`RAG_ASR_AISHELL_HOTWORD_DIR`、`RAG_ASR_AISHELL2_HOTWORD_DIR`、`RAG_ASR_AISHELL3_HOTWORD_DIR`、`RAG_ASR_MAGICDATA_HOTWORD_DIR`、`RAG_ASR_THCHS30_HOTWORD_DIR`、`RAG_ASR_ZHVOICE_HOTWORD_DIR`。训练卡数默认从 `CUDA_VISIBLE_DEVICES` 推导，可用 `RAG_ASR_NUM_GPUS` 覆盖。

`evaluation/` 下的压测脚本默认读取 `configs/serve.yaml` 或 `RAG_ASR_CONFIG`，并提供 `--config`、`--base-model-path`、`--adapter-ckpt`、`--hotword-pool-file`、`--cache-dir`、`--device` 做临时覆盖。

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

冒烟脚本与可运行示例在 [examples/](../examples)，详见 [examples/README.md](../examples/README.md)：

| 脚本 | 语义 |
|------|------|
| `examples/triton_client_example.py` | v1 Triton 单条音频最小调用；功能已被 `triton_hotword_client.py infer` 覆盖 |
| `examples/triton_v2_batch_example.py` | v2 显式 batch 协议示例 |
| `examples/triton_recall_check.py` | 对 `examples/` 样例做 Triton 端到端 recall 验证 |
| `examples/vllm_encoder_bypass.py` | 单条音频对比 vLLM 原始 encoder 与 Triton `PROJECTOR_OUT` → vLLM `audio_embeds` bypass |

## 数据集级评测和压测

评测与压测在 [evaluation/](../evaluation)，详见 [evaluation/README.md](../evaluation/README.md)；公共 bypass 协议在 `src/rag_asr/vllm_bypass.py`：

| 脚本 | 语义 |
|------|------|
| `evaluation/benchmark_vllm_encoder_bypass.py` | 批量对比纯 vLLM encoder 与 Triton bypass 的 CER 与时延 |
| `evaluation/benchmark_triton_vs_local.py` | v1 Triton 与本地 Python 的一致性和时延对比 |
| `evaluation/benchmark_triton_v2_batch.py` | v2 batch 协议压测 |

## 目录分层约定

- `scripts/`：shell 运维入口与其薄实现/兼容垫片（训练、离线推理、服务启动、环境构建、本地调试）。
- `src/rag_asr/`：可安装、可复用、可单测的库逻辑。
- `examples/`：可独立运行的最小示例与冒烟脚本，以及示例数据；依赖在线服务。
- `evaluation/`：数据集级离线评测与压测，产物写到 `var/`；依赖在线服务。
- `tests/`：纯 pytest 单测，无网络、无本机服务依赖。

## 后续整理方向

- examples/ 与 evaluation/ 分层已落地；冒烟/示例脚本不再以 `test_` 前缀命名，避免被 pytest 误收集。
- `retrieve.py` 适合逐步内收到 `src/rag_asr/cli_retrieve.py`，让 `rag-asr-retrieve` 不再依赖脚本路径。
- `merge_hw_maps.py` 等兼容入口等 `infer.sh` 完全改用 console script 后再移动或删除。
