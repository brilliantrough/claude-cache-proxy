# 测试指南

## 📁 测试结构

```
tests/
├── anthropic/                    # Anthropic 代理测试
│   ├── test_anthropic_core_endpoints.py  # 核心端点测试
│   └── test_anthropic_concurrency.py     # 并发性能测试
└── openrouter/                   # OpenRouter 代理测试
    ├── test_openrouter_basic.py           # 基本连通性测试
    ├── test_openrouter_core_endpoints.py   # 核心端点测试
    └── test_openrouter_concurrency.py      # 并发性能测试
```

## 🚀 运行测试

```bash
# 运行所有测试
python test_suite.py

# 单独测试
python tests/anthropic/test_anthropic_concurrency.py
python tests/openrouter/test_openrouter_concurrency.py
```

## 📋 测试说明

- **核心端点测试** - 验证健康检查、模型列表、服务信息
- **并发性能测试** - 验证高并发处理能力（最重要）
- **不测试 `/v1/messages`** - 并发测试已覆盖实际使用场景

## 🔧 测试配置

确保服务正在运行：
```bash
# 终端1
python anthropic_proxy.py

# 终端2
python openrouter_proxy.py
```

## 💡 测试策略

- **开发阶段** - 运行核心端点测试
- **部署前** - 运行并发性能测试
- **完整验证** - 运行 `test_suite.py`