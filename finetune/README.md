# 模型微调训练模块

本模块用于自定义对话角色的LoRA微调训练，全程针对双核CPU、8G内存环境优化，可在老旧设备上稳定完成训练。

## 文件说明
| 文件 | 功能说明 |
|------|----------|
| `finetune.py` | 微调训练主脚本，两阶段训练，适配低配CPU优化，降低内存占用 |
| `merge.py` | 将训练好的LoRA权重合并到基础模型，生成完整可推理的Hugging Face格式模型 |
| `test_inference.py` | 命令行交互式推理测试脚本，验证微调后的模型效果 |

## 使用步骤
### 1. 准备工作
1.  准备基础模型：推荐使用 `Qwen/Qwen2.5-0.5B-Instruct` 轻量级开源模型，适配低配环境
2.  准备训练数据：按照 `data/custom_role_data_example.json` 的格式，准备自定义角色的训练数据，命名为 `custom_role_data.json` 放入 `data/` 目录
3.  修改脚本中的路径配置，匹配你的本地文件路径

### 2. 开始微调训练
```bash
python finetune.py
```

脚本会自动完成数据清洗、两阶段训练，最终输出 LoRA 权重文件。
### 3. 合并 LoRA 权重
训练完成后，运行合并脚本，生成完整的可推理模型：
```bash
python merge.py
```

### 4. 命令行测试模型效果
```bash
python test_inference.py
```

### 5. 转换为 GGUF 格式（用于 webui / 移动端部署）
合并后的 Hugging Face 格式模型，需转换为 GGUF 格式才能用于 llama.cpp 推理，步骤如下：
克隆 llama.cpp 仓库
```bash
git clone https://github.com/ggerganov/llama.cpp
```
  安装转换依赖
```bash
pip install -r llama.cpp/requirements.txt
```

  执行转换（将路径替换为你的合并后模型路径）
```bash
python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile ./custom_role_model_q8_0.gguf --outtype q8_0
```

  转换完成后，将生成的 GGUF 文件放入 webui/model/ 目录，即可用于网页服务部署
