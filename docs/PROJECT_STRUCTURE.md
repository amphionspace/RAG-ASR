# 项目组织说明

## 问题复述

这个仓库真正要解决的问题不是“目录看起来是否整齐”，而是让训练、离线检索、在线服务和 Triton 部署在同一个项目里保持清晰边界。

## 关键假设

- 高风险：Triton 部署目录有固定约定，不能随意改 `rag_asr_retrieve/1/model.py` 的相对层级。
- 中风险：当前脚本依赖多处本机绝对路径，重组时应先集中配置，再移动目录。
- 低风险：`build/`、`*.egg-info`、`exp/`、`_retrieve_cache/` 属于本地生成物，源码结构评审时不应把它们当成维护对象。

## 当前结构

```text
RAG-ASR/
├── src/rag_asr/       # Python 包：模型、训练、推理、服务核心
├── scripts/           # Shell 和 CLI 脚本：训练、推理、服务启动、测试
├── triton/            # Triton model repository
├── docs/              # 文档
├── examples/          # 冒烟测试样例
├── checkpoints/       # 权重说明和本机权重入口
├── configs/           # 本机配置模板
└── exp/, build/       # 本地生成物，已忽略
```

## Triton 目录规则

`triton/rag_asr_retrieve/1/` 是 Triton 标准模型版本目录：

```text
triton/
└── rag_asr_retrieve/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

`1` 表示模型版本 1。这个目录不是 Python 包，也不是业务模块。只要继续使用 Triton Python Backend，就应该保留这种结构。

## 推荐演进顺序

1. 保持当前 `src/`、`scripts/`、`triton/` 三层可运行结构。
2. 把本机路径集中到 `configs/serve.yaml`，减少 `config.pbtxt`、`serve_http.sh`、`infer.sh` 之间的漂移。
3. 将运行产物逐步收敛到 `var/`，例如 `var/cache/` 和 `var/exp/`。
4. 按生命周期拆分脚本目录：`scripts/train/`、`scripts/infer/`、`scripts/serve/`、`scripts/env/`。
5. 在服务稳定后，再考虑把 `triton/` 整体迁到 `deployments/triton/`。
6. 最后拆分 Python 包内部大文件，例如 `dual_tower.py`、`infer.py` 和 `train.py`。

## 主要 trade-off

- 保留 `triton/` 根目录：最少改动，服务脚本不容易坏；代价是根目录仍暴露部署细节。
- 迁移到 `deployments/triton/`：语义更清晰；代价是需要同步修改启动脚本、文档、Triton 测试和部署手册。
- 先集中配置：短期收益最大，能降低路径硬编码风险；代价是需要维护 `.env` 或配置渲染逻辑。
- 直接深拆 Python 包：长期可维护性更好；代价是 import 路径、console script 和执行环境都要回归测试。
