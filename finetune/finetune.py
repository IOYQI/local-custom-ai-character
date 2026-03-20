import os
import shutil
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import json

# ====================== 环境配置（适配双核老CPU） ======================
torch.set_num_threads(2)  # 适配i3-2120 2核4线程
torch.set_grad_enabled(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ====================== 路径配置（使用前请修改） ======================
BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # 轻量级基础模型
TRAIN_DATA_PATH = "../data/custom_role_data.json"  # 你的训练数据路径
OUTPUT_DIR = "./lora_output"                       # 权重输出目录
MAX_SEQ_LENGTH = 512                                # 适配低内存环境

# ====================== 训练配置（老CPU可稳定运行） ======================
# 两阶段训练：先核心人设学习，再全量数据拟合
TRAIN_STAGES = [
    {"epochs": 3, "learning_rate": 1e-4, "data_filter": "core"},
    {"epochs": 5, "learning_rate": 5e-5, "data_filter": "full"}
]

# LoRA配置，平衡训练效果与内存占用
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    inference_mode=False,
)

# ====================== 工具函数 ======================
def backup_old_output():
    """备份旧的输出目录，避免文件丢失"""
    if os.path.exists(OUTPUT_DIR):
        backup_dir = f"{OUTPUT_DIR}_backup_{int(os.path.getmtime(OUTPUT_DIR))}"
        shutil.move(OUTPUT_DIR, backup_dir)
        print(f"已备份旧输出目录至 {backup_dir}")

def load_and_process_data(tokenizer, filter_type="full"):
    """加载并处理训练数据"""
    # 校验训练数据文件
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(
            f"训练数据未找到：{TRAIN_DATA_PATH}\n"
            f"请先按照 data/custom_role_data_example.json 的格式准备训练数据，放入data目录"
        )
    
    # 加载数据
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 数据清洗
    cleaned_data = []
    for item in raw_data:
        if not item.get("conversations"):
            continue
        # 过滤无效短数据
        valid = True
        for msg in item["conversations"]:
            if len(msg.get("value", "").strip()) < 2:
                valid = False
                break
        if valid:
            # 核心数据过滤（仅训练核心人设）
            if filter_type == "core" and not item.get("is_core", False):
                continue
            cleaned_data.append(item)
    
    # 数据去重
    unique_data = list({json.dumps(item, sort_keys=True): item for item in cleaned_data}.values())
    print(f"数据加载完成，有效训练数据量：{len(unique_data)} 条")

    # 格式化prompt，适配模型chat模板
    def format_func(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for conv in examples["conversations"]:
            messages = []
            for msg in conv:
                messages.append({"role": msg["from"], "content": msg["value"]})
            
            # 应用模型chat模板
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            tokenized = tokenizer(
                prompt,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # 标签处理：仅学习assistant回复部分，mask掉用户输入与padding
            input_ids = tokenized["input_ids"][0]
            labels = input_ids.clone()
            sep_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            user_end_pos = 0
            # 找到用户输入结束位置，mask掉
            for i in range(len(input_ids)-1, -1, -1):
                if input_ids[i] == sep_token_id and i > 2:
                    if input_ids[i-2] == tokenizer.convert_tokens_to_ids("user"):
                        user_end_pos = i+1
                        break
            labels[:user_end_pos] = -100
            labels[input_ids == tokenizer.pad_token_id] = -100

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(tokenized["attention_mask"][0])
            model_inputs["labels"].append(labels)
        
        return model_inputs

    dataset = Dataset.from_list(unique_data)
    processed_dataset = dataset.map(
        format_func,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1
    )
    processed_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return processed_dataset

# ====================== 主训练流程 ======================
def main():
    backup_old_output()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型和分词器
    print(f"正在加载基础模型：{BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"  # 因果语言模型规范，避免生成异常
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型，适配低内存环境
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    # 开启梯度检查点，降低内存占用
    model.gradient_checkpointing_enable()
    # 加载LoRA配置
    model = get_peft_model(model, LORA_CONFIG)
    print("可训练参数情况：")
    model.print_trainable_parameters()

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH
    )

    # 两阶段训练循环
    for stage_idx, stage in enumerate(TRAIN_STAGES):
        print(f"\n===== 开始第 {stage_idx+1} 阶段训练 =====")
        print(f"训练轮数：{stage['epochs']}，学习率：{stage['learning_rate']}")
        train_dataset = load_and_process_data(tokenizer, filter_type=stage["data_filter"])

        # 训练参数，适配老CPU
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=stage["epochs"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=stage["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="epoch",
            optim="adamw_torch",
            dataloader_num_workers=0,
            save_total_limit=2,
            report_to="none",
            disable_tqdm=False,
            fp16=False,  # 无GPU，关闭半精度
        )

        # 初始化训练器（直接使用 Trainer，无需 accelerate）
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        # 开始训练
        trainer.train()

    # 保存最终LoRA权重
    final_lora_path = os.path.join(OUTPUT_DIR, "final_lora")
    model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)
    print(f"\n训练全部完成！最终LoRA权重已保存至：{final_lora_path}")
    print("接下来可运行 merge.py 合并权重到基础模型")

if __name__ == "__main__":
    main()
