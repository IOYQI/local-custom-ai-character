import os
import subprocess
import time
import re
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ====================== 路径配置（使用前请修改） ======================
# 自动适配Windows和Termux环境
if os.name == "nt":
    # Windows环境配置（你的老电脑）
    LLAMA_CLI_PATH = "./llama.cpp/build/bin/llama-cli.exe"  # Windows下的llama-cli路径
    MODEL_PATH = "./model/custom_role_model_q8_0.gguf"     # 你的GGUF模型路径
else:
    # Termux/Android环境配置
    LLAMA_CLI_PATH = "/data/data/com.termux/files/home/llama/llama.cpp/build/bin/llama-cli"
    MODEL_PATH = "/data/data/com.termux/files/home/llama/models/custom_role_model_q8_0.gguf"

PROMPT_CACHE = "./prompt.cache"
HISTORY_FILE = "./chat_history.json"

# ====================== 环境检查 ======================
if not os.path.exists(LLAMA_CLI_PATH):
    print(f"警告：llama-cli 未找到：{LLAMA_CLI_PATH}")
    print("请先编译llama.cpp，并修改脚本中的LLAMA_CLI_PATH配置")
if not os.path.exists(MODEL_PATH):
    print(f"警告：模型文件未找到：{MODEL_PATH}")
    print("请将GGUF格式模型放入model目录，并修改脚本中的MODEL_PATH配置")

# ====================== 默认系统人设 ======================
DEFAULT_SYSTEM = (
    "你是一个友好的本地AI对话助手，拥有自然、口语化的说话风格。"
    "请用简洁、易懂的语言和用户对话，专注于回应用户的需求，保持对话流畅自然。"
)

# ====================== 对话历史管理 ======================
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(hist):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"历史记录保存失败: {e}")

history = load_history()
MAX_HISTORY_NUM = 10

# ====================== 路由配置 ======================
@app.route('/')
def index():
    return render_template('index.html', history=history, default_system=DEFAULT_SYSTEM)

@app.route('/send', methods=['POST'])
def send_message():
    start_time = time.time()
    data = request.get_json()
    user_msg = data.get('msg', '').strip()
    print(f"[{time.strftime('%H:%M:%S')}] 收到用户消息: {user_msg}")

    if not user_msg:
        return jsonify({"status": "error", "msg": "消息内容不能为空"})

    # 获取参数
    system_prompt = data.get('system', DEFAULT_SYSTEM)
    temperature = float(data.get('temp', 0.7))
    repeat_penalty = float(data.get('repeat_pen', 1.1))
    max_tokens = int(data.get('max_tokens', 64))
    ctx_window = 512  # 适配低内存环境

    # 构建prompt
    prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    for item in history[-5:]:
        prompt_parts.append(f"<|im_start|>user\n{item['user']}<|im_end|>\n")
        prompt_parts.append(f"<|im_start|>assistant\n{item['bot']}<|im_end|>\n")
    prompt_parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")
    final_prompt = "".join(prompt_parts)

    # 构建llama-cli命令，适配老CPU
    cmd = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "--temp", str(temperature),
        "--repeat-penalty", str(repeat_penalty),
        "-n", str(max_tokens),
        "-c", str(ctx_window),
        "-t", "1",  # 单线程，适配老CPU避免卡顿
        "-p", final_prompt,
    ]
    if os.path.exists(PROMPT_CACHE):
        cmd += ["--prompt-cache", PROMPT_CACHE]

    # 执行推理
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start_time
        print(f"推理完成，耗时 {elapsed:.2f} 秒")

        if result.returncode != 0:
            err_msg = result.stderr.strip() or "模型推理发生未知错误"
            return jsonify({"status": "error", "msg": f"执行失败: {err_msg}"})

        # 清理输出
        raw_output = result.stdout.strip()
        clean_patterns = [r'<\|im_end\|>', r'<\|endoftext\|>', r'</s>', r'\[end of text\]', r'^\s+|\s+$']
        for pattern in clean_patterns:
            raw_output = re.sub(pattern, '', raw_output, flags=re.MULTILINE)
        bot_reply = raw_output.strip()

        if not bot_reply:
            bot_reply = "我暂时无法回答这个问题，请换个话题试试吧。"

    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "msg": "生成超时，请稍后重试"})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"发生意外错误: {str(e)}"})

    # 保存历史
    history.append({"user": user_msg, "bot": bot_reply})
    if len(history) > MAX_HISTORY_NUM:
        history.pop(0)
    save_history(history)

    return jsonify({"status": "success", "bot": bot_reply})

@app.route('/delete/<int:idx>', methods=['POST'])
def delete_message(idx):
    global history
    if 0 <= idx < len(history):
        history.pop(idx)
        save_history(history)
    return jsonify({"status": "success"})

@app.route('/clear', methods=['POST'])
def clear_all_history():
    global history
    history = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return jsonify({"status": "success"})

# ====================== 常见问题FAQ ======================
"""
常见问题解决：
1.  服务启动后，http://localhost:8088 无法访问：
    - 检查防火墙是否开放8088端口，关闭Windows防火墙或添加端口放行规则
    - 检查llama-cli和模型文件路径是否正确，服务是否正常启动
2.  同局域网设备无法访问（如http://192.168.1.100:8088）：
    - 确保运行服务的设备和访问设备在同一WiFi/局域网下
    - 查看运行设备的局域网IP，确保IP地址正确
    - 关闭运行设备的防火墙，或放行8088端口
3.  模型推理失败：
    - 检查模型是否为GGUF格式，与llama.cpp版本兼容
    - 检查模型路径是否正确，文件是否完整
4.  生成速度慢：
    - 降低模型量化等级（如q4_0）、减小上下文窗口、调整线程数，适配设备性能
"""

# ====================== 启动服务 ======================
if __name__ == '__main__':
    # 启动前清理旧缓存
    if os.path.exists(PROMPT_CACHE):
        os.remove(PROMPT_CACHE)
        print("已清理旧缓存文件")
    # 0.0.0.0允许同局域网访问，端口8088
    app.run(host='0.0.0.0', port=8088, debug=False, threaded=True)
