#!/bin/bash
# 模型下载脚本

set -e

# 默认参数
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
CACHE_DIR="${CACHE_DIR:-./model_cache}"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "下载模型"
echo "=========================================="
echo "模型: $MODEL"
echo "缓存目录: $CACHE_DIR"
echo "=========================================="

python scripts/download_model.py \
    --model "$MODEL" \
    --cache-dir "$CACHE_DIR"

echo ""
echo "下载完成！"
