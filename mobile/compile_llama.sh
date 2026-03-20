```bash
#!/bin/bash
# Termux环境一键编译llama.cpp，适配老旧安卓手机
set -e

# 创建工作目录
cd ~
if [ ! -d llama ]; then
    mkdir llama
fi
cd llama

# 下载指定版本源码（稳定兼容版）
if [ ! -f llama.cpp-b3814.tar.gz ]; then
    wget https://github.com/ggerganov/llama.cpp/archive/refs/tags/b3814.tar.gz -O llama.cpp-b3814.tar.gz
fi

# 解压源码
tar -xzf llama.cpp-b3814.tar.gz
mv llama.cpp-b3814 llama.cpp
cd llama.cpp

# 适配安卓环境的编译配置
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_FLAGS="-include errno.h -D_POSIX_C_SOURCE=200112L -DPOSIX_MADV_WILLNEED=0 -DPOSIX_MADV_RANDOM=0 -D'posix_madvise(a,b,c)=0'" \
  -DLLAMA_DISABLE_MMAP=ON

# 单线程编译，避免小内存设备卡死
make -j1

echo "✅ 编译完成！可执行文件位于 build/bin/llama-cli"
