# 移动端离线部署指南

本项目已完成老旧安卓手机的端侧适配优化，可在3GB内存的老旧安卓手机上实现完全离线的AI对话服务，以下是完整部署步骤与适配优化说明。

## 适配设备
- 测试设备：老旧安卓手机（3GB内存、安卓6.0+系统）
- 运行环境：Termux 安卓终端环境
- 适配模型：Qwen2.5-0.5B-Instruct 微调后的GGUF量化模型

## 核心适配优化
1.  单线程推理优化，适配移动端低性能处理器，避免卡顿
2.  限制上下文窗口为512，降低内存占用，适配小内存设备
3.  禁用MMAP优化，解决安卓低版本系统兼容性问题
4.  优化前端界面，适配移动端屏幕尺寸，交互体验流畅

## 完整部署步骤
### 1. 环境准备
1.  安卓手机安装 Termux 终端
2.  打开Termux，执行以下命令更新环境并安装依赖：
    ```bash
    pkg update && pkg upgrade -y
    pkg install -y python python-pip git cmake make clang
    ```
3.  配置 Python 国内镜像源，提升安装速度：
    ```bash
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### 2. 编译 llama.cpp
一键执行编译脚本（本目录下的`compile_llama.sh`）：
```bash
sh compile_llama.sh
 ```
 
编译完成后， llama-cli  可执行文件会生成在  ~/llama/llama.cpp/build/bin/  目录。
 
### 3. 部署项目与模型
 
1. 克隆本仓库到 Termux 环境：
```bash 
git clone <你的GitHub仓库地址>
cd local-custom-llm-full-project
 ```

2. 安装 Python 依赖：
```bash
pip install -r requirements.txt
 ```

3. 将电脑上转换好的 GGUF 模型文件，放入手机的  ~/llama/models/  目录。
 
### 4. 启动服务
 
1. 进入 webui 目录：
```bash
cd webui
 ```

2. 启动后端服务：
```bash  
python app.py
 ```

3. 服务启动后，在手机浏览器打开  http://localhost:8088  即可开始离线对话；同局域网设备也可通过手机的局域网 IP 访问服务。
 
### 5. 优化参数建议
 
针对老旧手机的低配置环境，建议使用以下推理参数，保证流畅运行：
 
- 温度（temperature）：0.7
- 重复惩罚：1.1
- 最大生成长度：64
- 上下文窗口：512
- 推理线程数：1-2