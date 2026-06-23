# RAG-ASR 在线检索服务

将双塔热词检索封装为在线服务：输入一段音频，返回 **热词列表** 与 **帧级 projector 输出**（供下游 ASR 偏置使用）。

## 能力说明

| 输出 | 含义 | 形状（示例） |
|------|------|----------------|
| `WORD_LIST` / `word_list` | 与词池做相似度检索后的 top-K 热词 | JSON 字符串列表，默认 K=50 |
| `PROJECTOR_OUT` / `projector_out` | audio encoder → projector 的**帧级**特征 | `(T', D_proj)`，如 `(106, 2048)` |
| `PROJECTOR_LEN` / `projector_len` | 有效帧数 `T'` | 标量 |

说明：

- **会返回**：帧级 projector 输出（`PROJECTOR_OUT`），即 audio 侧中间表示。
- **不返回**：最终 512 维整句检索向量（仅内部用于计算 `word_list`）。
- 输入音频建议 **16 kHz 单声道 float32**；其他采样率会在服务内重采样。

默认模型与 `scripts/infer.sh` 一致：

- 基座：`checkpoints/base/amphion_1.7b_merged`（`config.model_type=qwen3_asr`，目录名沿用历史命名）
- Adapter：`checkpoints/adapters/amphion-1.7b_retrieval_v1.2/best_adapter.pt`
- 词池：`/chenmingjie/lx/data/hotword/zh/zh-10k.txt`
- 配置模板：`configs/serve.env.example`

---

## 两种部署方式

```
                    ┌─────────────────────────────────────┐
  客户端            │  RAGASRRetriever (rag_asr/serve.py)  │
       │            └─────────────────────────────────────┘
       │                          ▲
       ├─ Triton HTTP/gRPC ───────┤  start_triton.sh
       │   :8000 / :8001          │
       │                          │
       └─ FastAPI /retrieve ──────┘  serve_http.sh
           :8080
```

| 方式 | 启动脚本 | 端口 | 适用场景 |
|------|----------|------|----------|
| **Triton（推荐生产）** | `scripts/start_triton.sh` | HTTP 8000，gRPC 8001 | 集群部署、标准推理协议 |
| **HTTP 直连（调试）** | `scripts/serve_http.sh` | 8080 | 本地快速验证，不依赖 Triton |

---

## Triton 目录为什么有 `1/`

`triton/rag_asr_retrieve/` 是 Triton model repository 中的一个模型目录。Triton 要求模型实现放在整数版本目录下：

```text
triton/
└── rag_asr_retrieve/       # 模型名，对应 config.pbtxt 中的 name
    ├── config.pbtxt        # 输入输出、backend、运行参数
    └── 1/                  # 模型版本 1
        └── model.py        # Python Backend 入口
```

因此 `1/` 不是业务目录，也不是 Python 包层级；它表示 Triton 模型版本。未来如果输入输出协议发生不兼容变化，可以新增 `2/model.py`，保留 `1/` 用于灰度或回滚。只更换 adapter、词池或路径参数时，通常不需要新增版本，更新 `config.pbtxt` 后重启即可。

---

## 环境准备

### 1. 开发 / 启动 Triton 用：`triton`

从 `vllm` clone，已配置 `tritonserver` 路径与 `tritonclient`：

```bash
conda activate triton
```

`conda activate triton` 时会自动设置：

- `TRITONSERVER_ROOT=/ai_sds_wuzz/MODELS/tritonserver-2.64.0/tritonserver`
- CUDA / DCGM / extra-libs 等 `LD_LIBRARY_PATH`

### 2. Triton Python Backend 执行环境：`triton-exec`

模型推理实际跑在此环境中（与 `vllm`、`lx` 等环境隔离）：

```bash
bash scripts/build_triton_exec_env.sh
```

脚本会：

- 创建 `triton-exec`（Python 3.12）
- 安装 `torch==2.10.0+cu128`、`transformers==4.57.6`、`librosa` 等
- 以非 editable 方式安装 `rag-asr`
- 写入 Triton 所需的 `bin/activate`
- 可选：打包 `triton-exec-env.tar.gz` 到共享盘（便携归档）

当前 `config.pbtxt` 直接使用 live 环境：

`/ai_sds_wuzz/MODELS/miniconda3/envs/triton-exec`

### 3. 首次启动前的系统 symlink

Triton 自带的 Python stub 硬编码前缀 `/opt/pyenv_build/versions/3.12.3`。`start_triton.sh` 会自动创建：

```text
/opt/pyenv_build/versions/3.12.3 -> triton-exec
```

若手动启动 `tritonserver`，需自行保证该 symlink 存在。

---

## 启动服务

### Triton

```bash
cd /chenmingjie/lx/RAG-ASR
conda activate triton
bash scripts/start_triton.sh
```

成功时日志中应出现：

- `rag_asr_retrieve | 1 | READY`
- `Started HTTPService at 0.0.0.0:8000`

健康检查：

```bash
curl http://localhost:8000/v2/health/ready
# 200
```

### HTTP 调试服务

```bash
conda activate triton
bash scripts/serve_http.sh
# 或指定端口：PORT=9000 bash scripts/serve_http.sh
```

---

## 调用示例

### Triton HTTP 客户端

单条 wav（需已安装 `tritonclient`）：

```bash
conda activate triton
python scripts/triton_client_test.py \
  --wav examples/audio/cv_zh_33411896.wav \
  --url localhost:8000 \
  --top-k 50
```

对 `examples/` 全集做召回评测：

```bash
python scripts/test_triton_examples.py --url localhost:8000
```

### HTTP 调试接口

```bash
curl -X POST http://localhost:8080/retrieve \
  -F "file=@examples/audio/cv_zh_33411896.wav" \
  -F "top_k=50" \
  -F "sample_rate=16000"
```

响应字段：`word_list`、`projector_len`、`projector_out`（二维数组）、`projector_dim`。

### Triton 模型接口（`rag_asr_retrieve`）

**输入**

| 名称 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `WAV` | FP32 | 是 | 一维波形 |
| `SAMPLE_RATE` | INT32 | 否 | 默认 16000 |
| `TOP_K` | INT32 | 否 | 默认 50（见 `config.pbtxt`） |

**输出**

| 名称 | 类型 | 说明 |
|------|------|------|
| `WORD_LIST` | STRING | JSON 数组字符串 |
| `PROJECTOR_OUT` | FP32 | `(T', D_proj)` |
| `PROJECTOR_LEN` | INT32 | 有效帧数 |

模型仓库：`triton/rag_asr_retrieve/`（`config.pbtxt` + `1/model.py`）。

---

## 测试数据

`examples/` 下含从 cv-zh 采样的 5 条音频及标注：

| 文件 | 说明 |
|------|------|
| `audio/*.wav` | 16 kHz 单声道 |
| `transcripts.tsv` | `id \t 转写` |
| `hotwords.tsv` | `id \t 热词 JSON` |
| `metadata.jsonl` | 汇总元数据 |

---

## 目录与脚本索引

```
RAG-ASR/
├── triton/rag_asr_retrieve/     # Triton 模型仓库
│   ├── config.pbtxt
│   └── 1/model.py
├── src/rag_asr/serve.py         # 核心检索逻辑
├── scripts/
│   ├── build_triton_exec_env.sh # 构建 triton-exec
│   ├── start_triton.sh          # 启动 Triton
│   ├── serve_http.sh            # 启动 HTTP 调试服务
│   ├── serve_http.py
│   ├── triton_client_test.py    # 单条 Triton 测试
│   └── test_triton_examples.py  # examples 批量测试 + recall
└── examples/                    # 样例音频与标注
```

---

## 常见问题

### `unable to find backend library for backend 'python'`

`tritonserver` 默认在 `/opt/tritonserver/backends` 找 backend。请使用 `scripts/start_triton.sh`（已加 `--backend-directory`），或手动指定：

```bash
tritonserver --backend-directory="${TRITONSERVER_ROOT}/backends" ...
```

### `Stub process is not healthy` / `failed to get the Python codec`

多为 Python stub 找不到 `/opt/pyenv_build/versions/3.12.3` 下的标准库。运行 `start_triton.sh` 创建 symlink，并确保 `triton-exec` 为 **Python 3.12**（非 3.13）。

### `CUDA driver version is insufficient`

Triton 自带 CUDA 13 runtime，与主机驱动 12.8 不匹配时，Triton 侧看不到 GPU。当前 `config.pbtxt` 使用 `KIND_CPU` 实例组，**PyTorch 仍通过 cu128 使用 GPU** 做推理。

### `Librosa is not installed`

在 `triton-exec` 中安装：`pip install librosa`，或重新运行 `build_triton_exec_env.sh`。

### Triton CUDA 与 conda `triton` 环境

- **`conda triton`**：跑 `tritonserver`、客户端、HTTP 调试。
- **`triton-exec`**：Triton Python Backend 内执行 `model.py` 的依赖环境。
- 不会影响 `vllm` 等其他 conda 环境。

---

## 修改配置

优先复制并维护本机配置模板：

```bash
cp configs/serve.env.example configs/serve.env
```

当前 Triton 仍直接读取 `triton/rag_asr_retrieve/config.pbtxt` 中的 `parameters`，需要同步以下字段：

- `base_model_path` / `adapter_ckpt` / `hotword_pool_file`
- `default_top_k`、`embed_dim`、`device`、`cache_dir`

修改后重启 Triton 即可加载新配置。后续建议把 `config.pbtxt` 改为由 `configs/serve.env` 渲染生成，避免路径在 `serve_http.sh`、`infer.sh` 和 Triton 配置中多处漂移。
