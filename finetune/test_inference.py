import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# ====================== 路径配置 ======================
MODEL_PATH = "./merged_model"  # 合并后的模型路径
LORA_PATH = None  # 若不合并模型，可直接填写LoRA路径：./lora_output/final_lora

# ====================== 默认系统人设（和训练时保持一致） ======================
DEFAULT_SYSTEM_PROMPT = """
你是一个自定义的对话角色，拥有自己独特的说话风格和性格设定。
请始终保持符合人设的对话方式，使用自然、口语化的表达，避免生硬的书面语。
请专注于和用户进行流畅、贴合人设的对话，不要偏离设定。
"""

# ====================== 环境配置（适配老CPU） ======================
torch.set_num_threads(2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cpu"

# ====================== 工具函数 ======================
def init_model():
    print("正在加载模型和分词器...")
    # 校验模型文件
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型未找到：{MODEL_PATH}，请先完成模型合并")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # 加载LoRA权重（如有）
    if LORA_PATH and os.path.exists(LORA_PATH):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
        print("已加载LoRA权重")

    model.eval()
    print("模型加载完成！")
    print("=== 对话指令 ===")
    print("输入 /exit 退出对话")
    print("输入 /clear 清空对话历史")
    print("输入 /del 删除上一条对话")
    print("==================\n")
    return model, tokenizer

# ====================== 主对话流程 ======================
def main():
    model, tokenizer = init_model()
    history = []
    MAX_HISTORY_LENGTH = 8  # 适配低内存环境，限制历史长度

    while True:
        user_input = input("用户：").strip()

        # 指令处理
        if user_input == "/exit":
            print("对话结束")
            break
        if user_input == "/clear":
            history = []
            print("已清空对话历史\n")
            continue
        if user_input == "/del":
            if len(history) >= 2:
                history = history[:-2]
                print("已删除上一条对话\n")
            else:
                history = []
                print("当前无对话可删除\n")
            continue
        if not user_input:
            print("输入不能为空\n")
            continue

        # 构建对话prompt
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        messages += history[-MAX_HISTORY_LENGTH:]
        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 生成回复，适配老CPU优化参数
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
            )

        # 解析输出
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
        print(f"AI：{response}\n")

        # 保存对话历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
