# Triton Batch V2 设计与验证报告

## 问题复述

本次目标是实现一个真正面向 batch 调用的 Triton v2 服务，同时不破坏 v1 单条 `WAV` 协议，并以现有本地逐条推理作为精度基准。

## 关键假设

- Qwen3 上游代码明确绕开 audio encoder batch 以保持精度，因此 packed encoder batch 必须先通过精度门禁。
- 现有 v1 `WAV` 输入是变长一维 tensor，不适合直接打开 Triton 原生 dynamic batching。
- v2 先使用显式 padded batch 请求，客户端主动组成 batch；Triton scheduler dynamic batching 作为后续阶段。

## v2 模型

新增模型目录：

```text
triton/rag_asr_retrieve_v2/
├── config.pbtxt
└── 1/model.py
```

模型名：

```text
rag_asr_retrieve_v2
```

## v2 输入输出协议

输入：

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| WAV_BATCH | FP32 | [B, T_max] | batch 内按最长 waveform 右侧补 0 |
| WAV_LEN | INT32 | [B] | 每条 waveform 的有效采样点数 |
| SAMPLE_RATE | INT32 | [B] 或 [1] | 每条采样率；单值时广播 |
| TOP_K | INT32 | [B] 或 [1] | 每条 top-K；单值时广播 |

输出：

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| PROJECTOR_OUT | FP32 | [B, T_proj_max, D_proj] | batch 内按最长 projector 帧补 0 |
| PROJECTOR_LEN | INT32 | [B] | 每条有效 projector 帧数 |
| WORD_LIST | STRING | [B] | 每项是 JSON 字符串数组 |

## 实现内容

1. `src/rag_asr/dual_tower.py`
   - 新增 `forward_with_projector_packed()`。
   - 新增 Qwen3 packed 实验路径 `_forward_with_projector_qwen3_packed()`。
   - 新增 `_qwen3_projector_lens()`，镜像 Qwen3 audio conv 输出长度计算。

2. `src/rag_asr/serve.py`
   - 新增 `infer_padded_batch()`，接收 `[B, T_max]` + `[B]` 的显式 batch 输入。
   - 新增 `_results_from_features()`，复用 text pool 打分和 projector 输出切片逻辑。
   - `infer()`、`infer_many()` 保持兼容。

3. `triton/rag_asr_retrieve_v2/`
   - 新增 v2 Triton Python Backend。
   - v2 默认 `packed_audio=false`，即显式 batch 协议 + 内部安全逐条 audio path。
   - `packed_audio=true` 保留为实验开关，但不作为默认路径。

4. `scripts/`
   - 新增 `triton_v2_client_test.py`。
   - 新增 `benchmark_triton_v2_batch.py`。

## 精度门禁

安全 v2 路径：`packed_audio=false`

| batch size | word_list 一致 | PROJECTOR_LEN 一致 | PROJECTOR_OUT shape 一致 | 最大绝对误差 |
|------------|----------------|--------------------|--------------------------|--------------|
| 1 | true | true | true | 0.0 |
| 2 | true | true | true | 0.0 |
| 4 | true | true | true | 0.0 |
| 8 | true | true | true | 0.0 |
| 16 | true | true | true | 0.0 |

Qwen3 packed 实验路径：`packed_audio=true`

| batch size | word_list 一致 | PROJECTOR_LEN 一致 | PROJECTOR_OUT shape 一致 | 最大绝对误差 |
|------------|----------------|--------------------|--------------------------|--------------|
| 1 | true | true | true | 0.0 |
| 2 | false | true | true | 0.14501953125 |
| 4 | false | true | true | 0.1563720703125 |

结论：Qwen3 packed encoder batch 未通过精度门禁。v2 默认必须保持 `packed_audio=false`，否则会改变热词排序结果。

## 性能结果

测试命令：

```bash
source /ai_sds_wuzz/MODELS/miniconda3/etc/profile.d/conda.sh
conda activate triton
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/chenmingjie/lx/RAG-ASR/src \
  python scripts/benchmark_triton_v2_batch.py \
  --url localhost:8000 \
  --batch-sizes 1 2 4 8 16 \
  --repeats 5 \
  --warmup 1
```

Triton v2 safe path 时延：

| batch size | mean | p95 | max |
|------------|------|-----|-----|
| 1 | 15.44 ms | 16.94 ms | 16.94 ms |
| 2 | 31.20 ms | 34.78 ms | 34.78 ms |
| 4 | 60.31 ms | 61.38 ms | 61.38 ms |
| 8 | 118.99 ms | 122.87 ms | 122.87 ms |
| 16 | 232.70 ms | 240.38 ms | 240.38 ms |

结论：safe v2 的时延基本随 batch size 线性增长，因为内部仍使用逐条 Qwen3 audio path。它解决了 batch I/O 协议与结果对齐问题，但尚未提升主计算吞吐。

## 推导链

1. v2 显式 batch schema 可以稳定表达变长 waveform 和变长 projector 输出。
2. safe v2 与本地逐条推理最大误差为 0.0，满足精度门禁。
3. packed Qwen3 batch 在 batch size 2 起改变 `PROJECTOR_OUT` 并导致 `word_list` 不一致，不能作为默认实现。
4. safe v2 时延随 batch size 线性增长，证明当前性能瓶颈仍在 Qwen3 audio tower 逐条执行。
5. 因为 explicit batch 尚未带来主计算收益，暂不建议打开 Triton 原生 dynamic batching。

## 推荐方案与 trade-off

当前生产默认仍建议使用 `rag_asr_retrieve` v1。`rag_asr_retrieve_v2` 适合作为显式 batch API 或灰度模型，不建议无条件直接替换 v1。

当前可上线的是 `rag_asr_retrieve_v2` safe path：

- 优点：接口真正支持 batch 请求，输出 batch tensor，精度与 v1/local 单条完全一致。
- 代价：吞吐没有本质提升，适合客户端需要一次提交多条音频的场景，不适合作为吞吐优化终点。

v2 不能直接替换 v1 的原因：

- 模型名不同：`rag_asr_retrieve_v2`。
- 输入协议不同：v2 需要 `WAV_BATCH` 和 `WAV_LEN`。
- 输出协议不同：v2 返回 batch tensor，需要客户端按 `PROJECTOR_LEN` 拆分。
- v2 safe path 不提升吞吐，替换收益只在接口批量化，而不是性能。

不建议默认启用 packed Qwen3：

- 优点：batch size 4 时延约 31 ms，远低于 safe path 的 60 ms。
- 代价：热词列表不一致，最大 projector 误差约 0.15，违反精度门禁。

Triton 原生 dynamic batching 的下一步：

1. 保持 v2 safe path 作为正确性基线。
2. 单独开实验分支研究 Qwen3 audio tower 的 batch 精度问题。
3. 只有当 packed encoder 与逐条路径数值一致后，再打开：

```text
max_batch_size: 8
dynamic_batching {
  preferred_batch_size: [ 2, 4, 8 ]
  max_queue_delay_microseconds: 5000
}
```

## 已知未知

- 未定位 Qwen3 packed path 误差来自卷积 chunk、attention mask、RoPE/position 或 dtype 差异。
- 未测试更长音频和真实线上分布下的 batch padding 浪费。
- v2 当前模型与 v1 同时加载时会各自加载一份 retriever，显存约翻倍；生产可只加载目标模型或改共享进程架构。
