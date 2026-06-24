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
- Adapter：`checkpoints/base/amphion_1.7b_merged/hotword_adapter/best_adapter.pt`
- 词池：默认 `examples/hotword_pool.txt`，生产在 `configs/serve.yaml` 中配置 `retrieval.hotword_pool_file`
- 配置入口：`configs/serve.yaml`

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

## 当前推荐配置

生产默认使用 `rag_asr_retrieve`（v1 单条 `WAV` 协议）。

推荐理由：

- v1 协议与现有客户端兼容，输入一条音频返回一条热词列表和一条 projector 序列。
- v1/v2 safe path 的核心计算都走逐条 Qwen3 audio path；v2 不会带来吞吐提升。
- v2 只有在客户端天然需要一次提交多条音频时才有接口价值。

上游限流建议：

| 目标 | 建议值 |
|------|--------|
| 推荐 in-flight 并发 | 2 |
| 可接受 in-flight 并发 | 4 |
| 不建议超过 | 8 |
| 单实例稳定吞吐上限 | 约 70-72 req/s |
| 建议队列超时 | 约 200 ms |

超过并发 4 后，吞吐基本不再提升，只会增加排队时延。压测中并发 128 的吞吐仍约 71 req/s，但 p95 已超过 1.5s。

### v2 是否可以直接替换 v1

不能无条件直接替换。

`rag_asr_retrieve_v2` 使用不同模型名和不同输入输出协议：

- v1 输入：`WAV`、`SAMPLE_RATE`、`TOP_K`
- v2 输入：`WAV_BATCH`、`WAV_LEN`、`SAMPLE_RATE`、`TOP_K`
- v1 输出：单条 `PROJECTOR_OUT`、单条 `PROJECTOR_LEN`、单条 `WORD_LIST`
- v2 输出：batch 形式 `PROJECTOR_OUT`、`PROJECTOR_LEN`、`WORD_LIST`

可以使用 v2 的条件：

- 上游愿意改客户端，按 `[B, T_max]` padding 并传 `WAV_LEN`。
- 下游能消费 batch 输出，按 `PROJECTOR_LEN[i]` 截断每条 projector。
- 保持 `packed_audio=false`，不要默认启用 packed Qwen3。

不建议同时在生产加载 v1 和 v2，除非显存足够且确实要并行灰度。当前两个模型会各自加载一份 retriever/基座，显存约翻倍。生产更推荐只加载一个目标模型。

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

## 批处理能力

当前服务支持服务内核 micro-batch：`RAGASRRetriever.infer_many()` 可以一次处理多条音频，Triton Python Backend 会把同一次 `execute(requests)` 中的多个 request 聚合后调用 `infer_many()`。

当前未启用 Triton scheduler 原生 dynamic batching：

```text
max_batch_size: 0
```

原因是 `WAV` 是变长一维输入。直接打开 `max_batch_size > 0` 会要求客户端和服务端重新约定 padded batch 或 ragged 输入长度元数据。详细验证结果见 `docs/TRITON_INFERENCE_TEST_REPORT.md`。

新增 v2 显式 batch 模型：

- 模型名：`rag_asr_retrieve_v2`
- 输入：`WAV_BATCH`、`WAV_LEN`、`SAMPLE_RATE`、`TOP_K`
- 输出：`PROJECTOR_OUT`、`PROJECTOR_LEN`、`WORD_LIST`
- 默认：`packed_audio=false`，保证与 v1/local 单条推理数值一致

v2 设计与验证见 `docs/TRITON_BATCH_V2_REPORT.md`。

---

## 环境准备

### 1. 开发 / 启动 Triton 用：`triton`

需要一个安装了 `tritonserver` 与 `tritonclient` 的 conda 环境：

```bash
conda activate triton
```

若 `conda activate triton` 后没有自动设置 `TRITONSERVER_ROOT`，请在环境中显式配置它，或在 `configs/serve.yaml` 中设置 `triton.backend_dir`。

### 2. Triton Python Backend 执行环境：`triton-exec`

模型推理实际跑在此环境中（与 `vllm`、`lx` 等环境隔离）：

```bash
bash scripts/build_triton_exec_env.sh
```

脚本会：

- 创建 `triton-exec`（Python 3.12）
- 安装 `torch==2.10.0+cu128`、`transformers==4.57.6`、`librosa` 等
- 以非 editable 方式安装 `rag-asr`
- 在构建执行环境时准备 Triton 所需的 `bin/activate`
- 可选：打包 `triton-exec-env.tar.gz` 到共享盘（便携归档）

`configs/serve.yaml` 的 `triton.exec_env` 指向该执行环境；`start_triton.sh` 会把它渲染进运行时 `config.pbtxt`。

### 3. 首次启动前的 Python stub symlink

部分 Triton Python stub 会固定查找构建时的 Python 前缀。默认配置兼容常见前缀：

```text
/opt/pyenv_build/versions/3.12.3 -> triton-exec
```

`start_triton.sh` 会按 `configs/serve.yaml` 中的 `triton.python_stub_link` 自动创建 symlink；可用 `RAG_ASR_TRITON_PYTHON_STUB_LINK` 覆盖，或设为 `off` / `none` 跳过自动创建。若手动启动 `tritonserver`，需自行保证对应 symlink 存在。

---

## 启动服务

### Triton

```bash
cd RAG-ASR
# edit configs/serve.yaml for this machine
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

热词库管理与检索统一客户端：

```bash
# 显示已有热词库，支持分页和子串过滤
python scripts/triton_hotword_client.py --url localhost:8000 list --limit 20
python scripts/triton_hotword_client.py --url localhost:8000 list --query 北京 --limit 20

# 批量添加或从文件导入
python scripts/triton_hotword_client.py --url localhost:8000 add 北京烤鸭 上海迪士尼
python scripts/triton_hotword_client.py --url localhost:8000 import hotwords.txt
python scripts/triton_hotword_client.py --url localhost:8000 import hotwords.json

# 批量删除与从服务端配置文件重载
python scripts/triton_hotword_client.py --url localhost:8000 delete 北京烤鸭
python scripts/triton_hotword_client.py --url localhost:8000 reload

# 音频检索仍返回 projector 与热词召回结果
python scripts/triton_hotword_client.py --url localhost:8000 infer \
  --wav examples/audio/cv_zh_33411896.wav \
  --top-k 50
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
| `WAV` | FP32 | 检索必填 | 一维波形 |
| `SAMPLE_RATE` | INT32 | 否 | 默认 16000 |
| `TOP_K` | INT32 | 否 | 默认 50（见 `config.pbtxt`） |
| `ACTION` | STRING | 否 | 默认 `infer`；可选 `infer`、`list`、`add`、`delete`、`reload` |
| `HOTWORDS` | STRING | 管理写操作必填 | JSON 字符串或 JSON 数组，用于 `add` / `delete` |
| `QUERY` | STRING | 否 | `list` 的子串过滤 |
| `LIMIT` | INT32 | 否 | `list` 返回数量 |
| `OFFSET` | INT32 | 否 | `list` 分页偏移 |

**输出**

| 名称 | 类型 | 说明 |
|------|------|------|
| `WORD_LIST` | STRING | JSON 数组字符串 |
| `PROJECTOR_OUT` | FP32 | `(T', D_proj)` |
| `PROJECTOR_LEN` | INT32 | 有效帧数 |
| `STATUS` | STRING | 管理动作状态 |
| `MESSAGE` | STRING | 管理动作 JSON 摘要，包含新增、删除、重复、非法等统计 |
| `HOTWORD_COUNT` | INT32 | 当前热词库总数 |
| `HOTWORD_LIST` | STRING | `list` 或写操作影响到的热词 JSON 数组 |

模型仓库：`triton/rag_asr_retrieve/`（`config.pbtxt` + `1/model.py`）。

热词管理说明：

- `ACTION` 未传时保持旧客户端兼容，仍按音频检索执行。
- `add` 会在同一个 Triton 模型实例内用已加载的 `tokenizer + text_tower + adapter` 为新增热词生成 embedding，并立即加入 `_pool_embs_gpu`。
- `delete` 按规范化后的 dedupe key 删除，英文/拉丁默认大小写不敏感，中文保持词面 exact。
- `list` 显示当前服务内存里的已有热词库，可分页返回，避免一次性拉取大词库。
- `import` 是客户端命令：客户端读取本地 txt/json 后把热词数组发给 Triton；服务端不会读取客户端传来的任意文件路径。
- 管理操作会把规范化后的词库写回 `hotword_pool_file`，并覆盖刷新 text embedding cache，保证重启后词库一致。
- 当前热词状态属于单个 Triton Python backend 实例，生产应保持 `instance_group count: 1`；多实例/多机需要额外同步机制。

入库规则：

- 先做 `strip()` 和连续空白折叠，过滤空字符串。
- 参考 AmphionASR 的脚本感知长度规则：CJK/Thai 按字符计长，英文按词计长；默认中文至少 2 字，英文至少 1 词。
- 按 canonical key 去重，重复提交不会重复生成 embedding。
- 不做拼音、近音或语义级合并，这些属于后续二阶段 rerank 或 hard-negative 逻辑。

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
│   ├── triton_hotword_client.py # Triton 检索 + 热词管理客户端
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

优先复制并维护服务配置模板：

```bash
# edit configs/serve.yaml for this machine
```

`scripts/start_triton.sh` 会读取 `configs/serve.yaml`（或 `RAG_ASR_CONFIG` 指向的文件），渲染运行时 model repository 到 `var/triton_repo`，再启动 Triton。源码中的 `triton/` 目录只作为模板，不再直接维护机器相关路径。

常用字段：

- `model.base_model_path`：基座模型目录。
- `model.adapter_subdir` / `model.adapter_filename`：默认读取 `base_model_path/hotword_adapter/best_adapter.pt`。
- `model.adapter_ckpt`：可选旧式显式 adapter 路径，设置后优先级高于内置目录。
- `retrieval.hotword_pool_file`：热词池文件。
- `retrieval.default_top_k`、`retrieval.cache_dir`、`runtime.device`。
- `triton.exec_env`、`triton.http_port`、`triton.grpc_port`、`triton.rendered_model_repo`。

修改 `configs/serve.yaml` 后重启 Triton 即可生效。HTTP debug 服务 `scripts/serve_http.sh` 也读取同一份配置。
