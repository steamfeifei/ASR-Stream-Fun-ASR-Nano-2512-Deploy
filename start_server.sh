#!/bin/bash
#!/bin/bash
# 作者：凌封
# 来源：https://aibook.ren (AI全书)

# 设置启动后其他依赖的模型下载缓存目录，不设置默认会下载到这个目录：/root/.cache/modelscope
#export MODELSCOPE_CACHE=/your/custom/path

set -e

# 获取脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="$SCRIPT_DIR/models"

# 检查/激活虚拟环境
# 0. 尝试加载 Conda 初始化脚本 (兼容常见安装路径)
if [ -z "$CONDA_PREFIX" ] || [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    echo "当前未激活目标环境 (处于 base 或无环境)，尝试自动激活 'asr' 环境..."
    
    # 常见 Conda 安装路径
    CONDA_PATHS=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    )
    
    FOUND_CONDA=false
    for cpath in "${CONDA_PATHS[@]}"; do
        if [ -f "$cpath" ]; then
            source "$cpath"
            FOUND_CONDA=true
            break
        fi
    done
    
    if [ "$FOUND_CONDA" = true ]; then
        conda activate asr || echo "Warning: 'conda activate asr' failed."
    fi
fi

# 1. 再次检查环境
if [[ -n "$CONDA_PREFIX" ]] || [[ -n "$VIRTUAL_ENV" ]]; then
    echo "当前环境: ${CONDA_PREFIX:-$VIRTUAL_ENV}"
else
    # 2. 如果仍未激活，尝试寻找本地 venv
    if [ -d "$VENV_DIR" ]; then
        echo "激活本地 venv: $VENV_DIR"
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: 未检测到激活的虚拟环境，且本地 venv 目录 ($VENV_DIR) 不存在。"
        echo "解决办法:"
        echo "1. 请先手动激活环境: conda activate asr"
        echo "2. 或者运行 ./install.sh 创建本地 venv"
        exit 1
    fi
fi

# 模型路径 (对应 download_model.py 下载后的路径)
MODEL_PATH="$MODELS_DIR/FunAudioLLM/Fun-ASR-Nano-2512"

echo "=== 启动 FunASR WebSocket 服务 ==="
echo "模型路径: $MODEL_PATH"
echo "端口: 10095"
echo "模式: online (流式)"

# 启动服务
# 注意: 本地部署时，VAD 和 PUNC 模型通常也会预下载。
# 如果未指定 --vad_model 和 --punc_model，脚本将尝试从 ModelScope 自动下载默认模型。
python funasr_wss_server.py \
  --port 10095 \
  --asr_model_online "$MODEL_PATH" \
  --asr_model "$MODEL_PATH" \
  --device cuda

# 注意：如果需要 HTTPS/WSS，请配置 --certfile 和 --keyfile 指向有效的 SSL 证书
