#!/bin/bash

# Claude Proxy 启动脚本

echo "🚀 启动 Claude Proxy 服务器..."

# 检查是否存在.env文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件"
    echo "📝 正在从模板创建 .env 文件..."
    cp .env.example .env
    echo "✅ 已创建 .env 文件，请编辑并添加你的 Anthropic API Key"
    echo "💡 编辑命令: nano .env"
    exit 1
fi

# 检查API Key
if ! grep -q "ANTHROPIC_API_KEY=" .env || grep -q "your_api_key_here" .env; then
    echo "❌ 请在 .env 文件中设置你的 ANTHROPIC_API_KEY"
    echo "💡 编辑命令: nano .env"
    exit 1
fi

echo "✅ 配置检查通过"

# 激活虚拟环境并启动服务器
echo "🌟 启动服务器..."
source .venv/bin/activate
python anthropic_proxy.py