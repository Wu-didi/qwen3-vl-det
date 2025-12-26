#!/bin/bash
# 启动 FastAPI 服务

set -e

# 默认参数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-false}"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "启动交通设备异常检测 API 服务"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Reload: $RELOAD"
echo "=========================================="

if [ "$RELOAD" = "true" ]; then
    # 开发模式（热重载）
    uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload
else
    # 生产模式
    uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS"
fi
