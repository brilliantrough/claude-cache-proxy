# Claude Proxy - 高性能 API 代理服务器

一个支持 Anthropic 和 OpenRouter 的高性能 API 代理服务器，提供智能缓存和并发优化。

## ✨ 特性

- 🚀 **双代理支持** - Anthropic + OpenRouter (OpenAI 格式)
- 💾 **智能缓存控制** - 自动优化缓存策略
- 🔄 **完全兼容** - 无缝替换原生 API
- 📊 **性能优化** - 高并发处理能力
- 🛡️ **错误处理** - 完善的异常处理机制

## 🛠️ 快速开始

### 1. 安装环境

```bash
# 克隆项目
git clone <repository-url>
cd claude_proxy

# 使用 uv 安装依赖
pip install uv
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 2. 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置（添加你的 API Keys）
nano .env
```

### 3. 启动服务

```bash
# 启动 Anthropic 代理
python anthropic_proxy.py

# 启动 OpenRouter 代理（新终端）
python openrouter_proxy.py
```

## 📡 服务端点

### Anthropic 代理 (默认: http://localhost:9999)
- `POST /v1/messages` - Anthropic 消息接口
- `GET /v1/models` - 模型列表
- `GET /health` - 健康检查

### OpenRouter 代理 (默认: http://localhost:9998)
- `POST /v1/chat/completions` - OpenAI 格式聊天接口
- `GET /v1/models` - 模型列表
- `GET /health` - 健康检查

## 🧪 测试

```bash
# 运行所有测试
python test_suite.py

# 单独测试
python tests/anthropic/test_anthropic_concurrency.py  # Anthropic 并发测试
python tests/openrouter/test_openrouter_concurrency.py  # OpenRouter 并发测试
```

## 📁 项目结构

```
claude_proxy/
├── anthropic_proxy.py     # Anthropic 代理服务器
├── openrouter_proxy.py    # OpenRouter 代理服务器
├── tests/                 # 测试目录
│   ├── anthropic/        # Anthropic 测试
│   └── openrouter/        # OpenRouter 测试
├── pyproject.toml         # 项目配置
└── README.md              # 项目文档
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！