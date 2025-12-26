#!/bin/bash
# 启动 Gradio Web UI

set -e

# 默认参数
PORT="${PORT:-7860}"
SHARE="${SHARE:-false}"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "启动 Gradio Web UI"
echo "=========================================="
echo "Port: $PORT"
echo "Share: $SHARE"
echo "=========================================="

if [ "$SHARE" = "true" ]; then
    GRADIO_SERVER_PORT=$PORT python gradio_app.py --share
else
    GRADIO_SERVER_PORT=$PORT python gradio_app.py
fi
