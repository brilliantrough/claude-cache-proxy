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

OpenRouter 代理支持后端模型别名映射和 `-thinking` 后缀兼容。模型别名优先维护在 `openrouter_model_map.json`，后续新增模型通常只需要修改这个 JSON 文件并重启代理。

可直接传入别名模型：

```python
response = requests.post(
    "http://localhost:9998/v1/chat/completions",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "model": "claude-opus-4-6",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
```

也可在模型名后添加 `-thinking` 启用思考模式：

```python
response = requests.post(
    "http://localhost:9998/v1/chat/completions",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "model": "claude-opus-4-6-thinking",
        "messages": [{"role": "user", "content": "请先思考再回答"}]
    }
)
```

默认规则：

- `claude-opus-4-6` 会自动映射为 `anthropic/claude-opus-4.6`
- `claude-opus-4-6-thinking` 会自动映射并开启 reasoning
- `claude-opus-4-7` 可在 `openrouter_model_map.json` 中映射为 `anthropic/claude-opus-4.7`
- `claude-opus-4-7-thinking` 会先去掉 `-thinking`，再使用同一个 JSON 映射并开启 reasoning
- `OPENROUTER_REASONING_BUDGET` 默认值为 `30000`
- `OPENROUTER_THINKING_MAX_TOKENS` 默认值为 `80000`
- 当启用 `-thinking` 时，如果 `max_tokens` 小于 `80000`，代理会自动提升到 `80000`

模型映射加载优先级：

1. 代码内置默认映射
2. `OPENROUTER_MODEL_MAP_FILE` 指向的 JSON 文件，默认是 `openrouter_model_map.json`
3. `.env` 里的 `OPENROUTER_MODEL_MAP_JSON`，用于临时覆盖

`openrouter_model_map.json` 示例：

```json
{
  "claude-opus-4-7": "anthropic/claude-opus-4.7",
  "claude-opus-latest": "anthropic/claude-opus-4.7"
}
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
OPENROUTER_MODEL_MAP_FILE=openrouter_model_map.json
OPENROUTER_REASONING_BUDGET=30000
OPENROUTER_THINKING_MAX_TOKENS=80000
OPENROUTER_MODEL_MAP_JSON={"claude-opus-4-7":"anthropic/claude-opus-4.7"}
```
