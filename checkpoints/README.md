# Checkpoints

本目录存放 RAG-ASR 推理/训练所需的权重，脚本只引用此目录下的路径。

```
checkpoints/
├── base/
│   └── amphion_1.7b_merged/     # ASR 基座（HF 格式）
└── adapters/
    └── amphion-1.7b_retrieval_v1.2/
        └── best_adapter.pt      # 双塔 adapter（~11MB，已复制）
```

## 基座模型

`base/amphion_1.7b_merged` 当前为软链接，已指向：

```
/chenmingjie/lx/AmphionASR/exp/amphion_asr_1.7b_hw-vitw-ts/v0-20260616-001820/checkpoint-5000_copy_merged
```

软链接可直接用于 `bash scripts/infer.sh`，**不必先复制**。

若需把基座真正拷进本项目（去掉软链接），先删链接再 `cp -a`：

```bash
cd /chenmingjie/lx/RAG-ASR
rm checkpoints/base/amphion_1.7b_merged
cp -a /chenmingjie/lx/AmphionASR/exp/amphion_asr_1.7b_hw-vitw-ts/v0-20260616-001820/checkpoint-5000_copy_merged \
      checkpoints/base/amphion_1.7b_merged
```

注意：路径里不能有 `...`，必须写完整目录名。

## Adapter

训练新 adapter 后放到 `checkpoints/adapters/<run_name>/best_adapter.pt`，并更新 `scripts/infer.sh` 中的 `ADAPTER` 路径。
