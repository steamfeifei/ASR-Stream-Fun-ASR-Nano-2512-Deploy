#!/bin/bash
#!/bin/bash
# 作者：凌封
# 来源：https://aibook.ren (AI全书)

# 获取当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 项目根目录 (假设 deploy/linux 在项目根目录下)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== 开始安装 Fun-ASR 环境 ==="
echo "工作目录: $SCRIPT_DIR"

# 抑制 pip 在 root 用户下运行的警告 (在 Docker/服务器环境中通常是安全的)
export PIP_ROOT_USER_ACTION=ignore

# 1. 检查/创建 Python 虚拟环境
# 如果已经在 Conda 环境或 VirtualEnv 中，则跳过创建
if [[ -n "$CONDA_PREFIX" ]] || [[ -n "$VIRTUAL_ENV" ]]; then
    echo "检测到当前已在虚拟环境中: ${CONDA_PREFIX:-$VIRTUAL_ENV}"
    echo "跳过 venv 创建，直接使用当前环境。"
    # 设置 VENV_DIR 为空，标记不使用本地 venv
    USE_LOCAL_VENV=false
else
    if [ ! -d "$VENV_DIR" ]; then
        echo "正在创建虚拟环境: $VENV_DIR ..."
        # 尝试使用 python3 或 python
        if command -v python3 &> /dev/null; then
            python3 -m venv "$VENV_DIR"
        else
            python -m venv "$VENV_DIR"
        fi
    else
        echo "虚拟环境已存在。"
    fi
    USE_LOCAL_VENV=true
fi

# 激活虚拟环境 (仅当使用本地 venv 时)
if [ "$USE_LOCAL_VENV" = true ]; then
    source "$VENV_DIR/bin/activate"
fi

# 2. 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 3. 安装 PyTorch
# 注意：您的服务器是 CUDA 12.4，我们安装兼容 CUDA 12.x 的 PyTorch (通常 cu121 版本兼容性最好)
echo "正在安装 PyTorch (CUDA 12.x)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装 FunASR, ModelScope 和 WebSocket 依赖
# 注意: Fun-ASR-Nano 需要 transformers 库
echo "正在安装 FunASR, ModelScope 和 websockets..."
pip install funasr modelscope websockets transformers sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 下载官方 WebSocket 客户端测试脚本
echo "正在下载 FunASR WebSocket 客户端脚本..."
# funasr_wss_server.py 已包含在本项目中，无需下载

if [ -f "funasr_wss_client.py" ]; then
    echo "funasr_wss_client.py 已存在，跳过下载"
else
    curl -O https://raw.githubusercontent.com/modelscope/FunASR/main/runtime/python/websocket/funasr_wss_client.py
fi

echo "=== 安装完成 ==="
echo "请使用以下命令激活环境："
echo "source $VENV_DIR/bin/activate"

# ==========================================
# 人工手动安装步骤说明 (仅供参考)
# ==========================================
# 如果您希望手动一步步执行安装，请参考以下命令：
#
# 1. 安装ffmpeg
# sudo apt update && sudo apt install ffmpeg
#
# 2. 进入部署目录
# cd /data/asr
#
# 3. 创建 Python 虚拟环境
# (推荐使用 Conda 创建 Python 3.10 环境，因为系统默认 Python 3.13 可能存在兼容性问题)
# conda create -n asr python=3.10 -y
# conda activate asr
#
# 或者使用原生 venv (如果 Python 版本 < 3.13)
# python3 -m venv venv
# source venv/bin/activate
#
# 4. 升级 pip
# pip install --upgrade pip
#
# 5. 安装 PyTorch (根据您的 CUDA 版本选择，这里适配 CUDA 12.x)
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# 6. 安装 FunASR 及相关依赖
# pip install funasr modelscope websockets transformers sentencepiece
#
# 7. (可选) 下载客户端测试脚本
# curl -O https://raw.githubusercontent.com/modelscope/FunASR/main/runtime/python/websocket/funasr_wss_client.py
#
# 安装完成后，可以使用 'python download_model.py' 下载模型
# 注意: 该脚本会自动下载 FunASR 和 dependent Qwen3 模型，并处理好目录结构。
