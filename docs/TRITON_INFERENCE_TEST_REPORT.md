# Triton 推理测试报告

## 问题复述

本次测试要验证两件事：修改后的服务是否能在服务内核聚合多请求做批量推理，以及 Triton 服务输出是否与本地 Python 推理保持数值一致。

## 关键假设

- 高风险：当前 `WAV` 是变长一维输入，未直接打开 Triton scheduler 的 `max_batch_size > 0`，避免破坏现有客户端协议。
- 中风险：Qwen3 audio tower 内部仍按 utterance 循环处理，因此本地 batch API 的收益不等同于真正 encoder batch 加速。
- 低风险：测试使用 `examples/` 下 5 条中文样例，覆盖服务 I/O 与 projector 输出一致性，但不代表全量线上分布。

## 修改内容

1. `src/rag_asr/serve.py` 新增 `RAGASRRetriever.infer_many()`，支持多条 waveform 统一提特征、padding 后调用 `forward_with_projector()`，再按 `PROJECTOR_LEN` 切回每条结果。
2. `RAGASRRetriever.infer()` 保持单条接口兼容，内部调用 `infer_many()`。
3. `triton/rag_asr_retrieve/1/model.py` 在一次 `execute(requests)` 中收集所有 request，调用一次 `infer_many()`，再按原顺序返回 `InferenceResponse`。
4. Triton backend 优先把当前仓库 `src/` 加入 `sys.path`，避免 `triton-exec` 中非 editable 安装的旧 `rag-asr` 包覆盖源码修改。
5. 新增 `evaluation/benchmark_triton_vs_local.py`，统一执行精度一致性、时延测试和小规模并发压测。

## 测试环境

```text
workspace: /chenmingjie/lx/RAG-ASR
conda env: triton
tritonserver: 2.64.0
Triton HTTP: localhost:8000
Triton gRPC: localhost:8001
CUDA_VISIBLE_DEVICES: 1
model: checkpoints/base/amphion_1.7b_merged
adapter: checkpoints/adapters/amphion-1.7b_retrieval_v1.2/best_adapter.pt
hotword pool: /chenmingjie/lx/data/hotword/zh/zh-10k.txt
top_k: 50
examples: 5
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/start_triton.sh
```

测试命令：

```bash
source /ai_sds_wuzz/MODELS/miniconda3/etc/profile.d/conda.sh
conda activate triton
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/chenmingjie/lx/RAG-ASR/src \
  python evaluation/benchmark_triton_vs_local.py \
  --url localhost:8000 \
  --pressure-concurrency 1 2 4 \
  --pressure-requests 8 \
  --warmup 1
```

## 精度结果

本地单条推理与本地 batch 推理：

| 指标 | 结果 |
|------|------|
| 样例数 | 5 |
| word_list 全部一致 | true |
| PROJECTOR_LEN 全部一致 | true |
| PROJECTOR_OUT shape 全部一致 | true |
| PROJECTOR_OUT 最大绝对误差 | 0.0 |

Triton 服务与本地 Python 推理：

| 指标 | 结果 |
|------|------|
| 样例数 | 5 |
| word_list 全部一致 | true |
| PROJECTOR_LEN 全部一致 | true |
| PROJECTOR_OUT shape 全部一致 | true |
| PROJECTOR_OUT 最大绝对误差 | 0.0 |

逐条 projector shape：

| id | shape |
|----|-------|
| cv_zh_33411896 | [106, 2048] |
| cv_zh_22213306 | [126, 2048] |
| cv_zh_22070948 | [76, 2048] |
| cv_zh_22153305 | [88, 2048] |
| cv_zh_19588558 | [71, 2048] |

## 时延结果

本地 Python：

| 模式 | 总耗时 |
|------|--------|
| 5 条逐条 infer | 82.90 ms |
| 5 条 infer_many | 424.65 ms |
| batch speedup | 0.195x |

解释：本地 batch 当前没有加速，主要因为当前 Qwen3 audio tower 路径在 `dual_tower.py` 内部仍逐 utterance 调用 audio tower；batch 化主要减少服务端重复编排，并为后续真正 encoder batch 化留接口。

Triton 单请求时延：

| 指标 | 数值 |
|------|------|
| count | 5 |
| mean | 19.71 ms |
| p50 | 19.74 ms |
| p90 | 24.76 ms |
| p95 | 24.76 ms |
| p99 | 24.76 ms |
| min | 15.15 ms |
| max | 24.76 ms |

## 压测结果

压测使用同一条短音频重复请求。第一轮每档 64 个请求，第二轮并发 64/128 每档 128 个请求。

| 并发 | 请求数 | wall time | 吞吐 | mean | p50 | p95 | max |
|------|--------|-----------|------|------|-----|-----|-----|
| 1 | 64 | 1.021 s | 62.71 req/s | 15.84 ms | 15.64 ms | 17.69 ms | 21.82 ms |
| 2 | 64 | 0.897 s | 71.33 req/s | 27.71 ms | 27.65 ms | 29.03 ms | 31.68 ms |
| 4 | 64 | 0.885 s | 72.33 req/s | 53.82 ms | 54.66 ms | 57.03 ms | 84.71 ms |
| 8 | 64 | 0.886 s | 72.25 req/s | 104.08 ms | 109.69 ms | 113.02 ms | 151.04 ms |
| 16 | 64 | 0.915 s | 69.91 req/s | 192.40 ms | 226.72 ms | 233.90 ms | 300.42 ms |
| 32 | 64 | 0.882 s | 72.59 req/s | 326.15 ms | 410.18 ms | 438.89 ms | 477.56 ms |
| 64 | 128 | 1.806 s | 70.89 req/s | 645.36 ms | 768.03 ms | 900.95 ms | 960.15 ms |
| 128 | 128 | 1.797 s | 71.25 req/s | 828.39 ms | 853.26 ms | 1515.25 ms | 1608.74 ms |

结论：吞吐在并发 2-4 后已经饱和，长期稳定上限约为 70-72 req/s。继续拉高并发不会提升吞吐，只会增加排队时延；并发 128 时 p95 已超过 1.5s。

## 推导链

1. 单条 `infer()` 和 `infer_many()` 的 `word_list`、`PROJECTOR_LEN`、`PROJECTOR_OUT` 完全一致，说明 batch API 没有改变模型结果。
2. Triton 与本地 Python 的最大数值误差为 0.0，说明 Triton tensor I/O、JSON word list 编码和 projector 输出没有引入精度偏差。
3. 并发从 1 到 4 时吞吐到达约 72 req/s，之后继续拉到 128 并发吞吐不再提升，说明当前瓶颈更接近单实例模型执行/调度，而不是 HTTP 客户端。
4. 本地 batch 慢于逐条推理，说明当前 Qwen3 audio tower 还没有利用真正的多 utterance encoder batch；后续优化应落在 `AmphionAudioTower._forward_with_projector_qwen3()`。

## 推荐方案与 trade-off

当前修改适合作为第一阶段：结果无精度偏差，Triton backend 能聚合 `execute(requests)` 中的多请求，并且保留原有单条 WAV 客户端协议。

主要 trade-off：

- 保留 `max_batch_size: 0`：兼容现有变长 WAV 输入；代价是没有启用 Triton scheduler 原生 dynamic batching。
- 服务内核 `infer_many()`：为后续批量优化提供接口；代价是 Qwen3 当前路径仍因内部逐条 audio tower 调用而没有吞吐收益。
- 不像 vLLM：当前没有 continuous batching、显存预算调度或 token-level scheduler；最大承载量主要由单 Triton instance、模型显存、请求长度和 Python backend 调度决定。

## 真正支持 batch 的方案

真正支持 batch 需要同时解决三层问题：

1. 输入协议：把变长 `WAV` 改成 batchable 表示。可选方案是 `WAV_BATCH` 使用 padded 二维张量 `[B, T_max]`，另加 `WAV_LEN` 表示每条有效采样点；或者使用 Triton ragged batching，把多条一维 waveform 拼接并传长度。
2. Triton 调度：将 `config.pbtxt` 改为 `max_batch_size > 0`，并配置 `dynamic_batching` 的 `preferred_batch_size` 和 `max_queue_delay_microseconds`。这样 Triton scheduler 才会把不同客户端请求合并成一个 batch 调用 backend。
3. 模型内部：改造 Qwen3 audio tower 路径，避免 `AmphionAudioTower._forward_with_projector_qwen3()` 内部逐 utterance 调用 `self.audio_tower(...)`。如果模型底层只支持 packed single utterance，就需要先实现 packed batch 调用或在更低层支持 batch encoder。

建议分阶段做：

1. 第一阶段：保留现有单条 `WAV` 客户端协议，继续使用当前 micro-batch，用于稳定性验证。
2. 第二阶段：新增一个 v2 Triton 模型版本，例如 `triton/rag_asr_retrieve/2/`，输入改为 `WAV_BATCH` + `WAV_LEN`，输出改为 padded `PROJECTOR_OUT` + `PROJECTOR_LEN` + `WORD_LIST` batch。
3. 第三阶段：打开 `max_batch_size: 8` 或 `16` 与 `dynamic_batching`，用真实多客户端请求验证吞吐。
4. 第四阶段：如果吞吐仍不提升，继续下钻到 Qwen3 audio tower，做真正 encoder batch 化；否则 Triton 层 batch 只是减少 Python 调用开销，无法改变主计算瓶颈。

## 已知未知

- 未测试长音频、大词池、多 GPU instance 或真实线上音频分布。
- 未测试 `instance_group count > 1`，因为每个实例会加载一份 1.7B 模型，显存成本高。
- 未启用 Triton 原生 ragged dynamic batching；如果要打开，需要重新设计输入协议，例如显式传入 padded batch 或增加长度元数据。
