# Checkpoints

本目录存放 RAG-ASR 推理/训练所需的权重，脚本只引用此目录下的路径。

```
checkpoints/
├── base/
│   └── amphion_1.7b_merged/     # ASR 基座（HF 格式）
│       └── hotword_adapter/
│           └── best_adapter.pt  # 推荐部署布局：热词 adapter 随基座目录交付
└── adapters/                    # 训练/实验输出的旧式分离 adapter，可选保留
```

## 基座模型

`base/amphion_1.7b_merged` 可以是 HF checkpoint 目录，也可以是指向外部 checkpoint 的软链接。

若需把基座真正拷进本项目（去掉软链接），先删链接再复制：

```bash
cd RAG-ASR
rm checkpoints/base/amphion_1.7b_merged
cp -a /path/to/hf_checkpoint checkpoints/base/amphion_1.7b_merged
```

注意：路径里不能有 `...`，必须写完整目录名。

## Adapter

训练新 adapter 后，可以先保存在 `checkpoints/adapters/<run_name>/best_adapter.pt`。服务部署时推荐复制到基座目录内：

```bash
mkdir -p checkpoints/base/amphion_1.7b_merged/hotword_adapter
cp checkpoints/adapters/<run_name>/best_adapter.pt \
   checkpoints/base/amphion_1.7b_merged/hotword_adapter/best_adapter.pt
```

服务默认读取 `base_model_path/hotword_adapter/best_adapter.pt`。如需临时使用分离 adapter，可在 `configs/serve.yaml` 中显式设置 `model.adapter_ckpt`。
