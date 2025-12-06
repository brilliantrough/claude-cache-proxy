# 使用指南

## 🚀 快速启动

### 1. 环境配置

```bash
# 安装 uv 并创建环境
pip install uv
uv sync
source .venv/bin/activate

# 配置 API Keys
cp .env.example .env
nano .env  # 添加你的 ANTHROPIC_API_KEY 和 OPENROUTER_API_KEY
```

### 2. 启动服务

```bash
# 终端1: 启动 Anthropic 代理 (端口 9999)
python anthropic_proxy.py

# 终端2: 启动 OpenRouter 代理 (端口 9998)
python openrouter_proxy.py
```

### 3. 使用代理

#### Anthropic 代理 (localhost:9999)
```python
import anthropic

client = anthropic.Anthropic(
    api_key="any-key",  # 代理会替换
    base_url="http://localhost:9999"
)

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### OpenRouter 代理 (localhost:9998)
```python
import requests

response = requests.post(
    "http://localhost:9998/v1/chat/completions",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "model": "anthropic/claude-4.5-sonnet",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
```

## 🧪 测试验证

```bash
# 运行所有测试
python test_suite.py

# 或单独测试
python tests/anthropic/test_anthropic_core_endpoints.py
python tests/openrouter/test_openrouter_core_endpoints.py
```

## 📋 API 端点

| 代理 | 端点 | 功能 |
|------|------|------|
| Anthropic | `/v1/messages` | 消息接口 |
| Anthropic | `/v1/models` | 模型列表 |
| Anthropic | `/health` | 健康检查 |
| OpenRouter | `/v1/chat/completions` | 聊天接口 |
| OpenRouter | `/v1/models` | 模型列表 |
| OpenRouter | `/health` | 健康检查 |

## 🔧 配置说明

`.env` 文件主要配置：
```env
# Anthropic
ANTHROPIC_API_KEY=your-key
PROXY_PORT=9999

# OpenRouter
OPENROUTER_API_KEY=your-key
OPENAI_PROXY_PORT=9998
```
