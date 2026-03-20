import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ====================== 路径配置（需和finetune.py保持一致） ======================
BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_WEIGHTS_PATH = "./lora_output/final_lora"
OUTPUT_MERGED_MODEL_PATH = "./merged_model"

# ====================== 环境配置（适配老CPU） ======================
torch.set_num_threads(2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== 工具函数 ======================
def backup_old_output():
    if os.path.exists(OUTPUT_MERGED_MODEL_PATH):
        backup_dir = f"{OUTPUT_MERGED_MODEL_PATH}_backup_{int(os.path.getmtime(OUTPUT_MERGED_MODEL_PATH))}"
        shutil.move(OUTPUT_MERGED_MODEL_PATH, backup_dir)
        print(f"已备份旧合并模型目录至 {backup_dir}")

# ====================== 主合并流程 ======================
def main():
    backup_old_output()
    os.makedirs(OUTPUT_MERGED_MODEL_PATH, exist_ok=True)

    # 校验文件
    if not os.path.exists(LORA_WEIGHTS_PATH):
        raise FileNotFoundError(f"LoRA权重未找到：{LORA_WEIGHTS_PATH}，请先完成训练")
    print("正在加载基础模型与LoRA权重...")

    # 加载基础模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 加载并合并LoRA权重
    lora_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    merged_model = lora_model.merge_and_unload()
    print("LoRA权重合并完成")

    # 保存合并后的完整模型，分片保存避免大文件问题
    merged_model.save_pretrained(
        OUTPUT_MERGED_MODEL_PATH,
        max_shard_size="500MB",
        safe_serialization=True
    )
    tokenizer.save_pretrained(OUTPUT_MERGED_MODEL_PATH)
    print(f"合并后的完整模型已保存至：{OUTPUT_MERGED_MODEL_PATH}")
    print("可参考finetune/README.md将此模型转换为GGUF格式，用于本地部署")

if __name__ == "__main__":
    main()
