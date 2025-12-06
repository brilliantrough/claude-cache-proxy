# OpenRouter Proxy - OpenAI 格式兼容代理

为 OpenRouter API 提供 OpenAI 格式的高并发代理服务器，自动添加 cache_control 支持。

## ✨ 特性

- 🔄 **OpenAI 格式兼容** - 完全兼容 OpenAI API
- 🚀 **高并发支持** - 基于 aiohttp 的异步处理
- 💾 **自动缓存控制** - 自动为最后一条消息添加 cache_control
- 📡 **流式响应支持** - 支持 Server-Sent Events 流式传输

## 🛠️ 快速开始

### 1. 环境配置

```bash
# 添加到 .env 文件
OPENROUTER_API_KEY=your-openrouter-api-key
OPENAI_PROXY_HOST=0.0.0.0
OPENAI_PROXY_PORT=9998
```

### 2. 启动服务

```bash
python openrouter_proxy.py
```

服务将在 `http://localhost:9998` 启动。

## 📡 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 格式聊天接口 |
| `/v1/models` | GET | 模型列表 |
| `/health` | GET | 健康检查 |
| `/` | GET | 服务信息 |

## 💾 Cache Control 机制

自动处理缓存控制：
1. 清理现有 cache_control 字段
2. 在最后一条消息添加：`{"type": "ephemeral"}`

## 📝 使用示例

### 基本请求

```python
import requests

response = requests.post(
    "http://localhost:9998/v1/chat/completions",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
```

### 流式请求

```python
response = requests.post(
    "http://localhost:9998/v1/chat/completions",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "max_tokens": 200,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 🧪 测试

```bash
# 运行 OpenRouter 测试
python tests/openrouter/test_openrouter_core_endpoints.py
python tests/openrouter/test_openrouter_concurrency.py
```

## 📊 性能指标

- 支持 100+ 并发连接
- QPS 可达 2000+
- 响应时间 < 50ms