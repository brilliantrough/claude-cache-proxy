#!/usr/bin/env python3
"""Anthropic cache_control 和 thinking 归一化测试。"""

import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def reload_anthropic_proxy():
    import anthropic_proxy

    return importlib.reload(anthropic_proxy)


def count_cache_control(data: Any) -> int:
    if isinstance(data, dict):
        return int("cache_control" in data) + sum(count_cache_control(value) for value in data.values())
    if isinstance(data, list):
        return sum(count_cache_control(item) for item in data)
    return 0


def test_cache_control_standardization_does_not_mutate_original() -> bool:
    anthropic_proxy = reload_anthropic_proxy()
    handler = anthropic_proxy.AnthropicRequestHandler("https://api.anthropic.com", "test-key", "5m")
    request_data = {
        "model": "claude-test",
        "system": [
            {
                "type": "text",
                "text": "system prompt",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    },
                    {"type": "text", "text": "world"},
                ],
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }
    original_request = copy.deepcopy(request_data)

    standardized = handler._standardize_cache_control(request_data)

    return (
        request_data == original_request
        and count_cache_control(standardized) == 1
        and standardized["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    )


def test_thinking_defaults_from_env_fallback() -> bool:
    os.environ.pop("ANTHROPIC_THINKING_BUDGET_TOKENS", None)
    os.environ.pop("ANTHROPIC_THINKING_MAX_TOKENS", None)
    anthropic_proxy = reload_anthropic_proxy()
    request_data = {
        "model": "claude-test-thinking",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }
    original_request = copy.deepcopy(request_data)

    normalized = anthropic_proxy.normalize_anthropic_model_and_thinking(request_data)

    return (
        request_data == original_request
        and normalized["model"] == "claude-test"
        and normalized["thinking"] == {"type": "enabled", "budget_tokens": 8192}
        and normalized["max_tokens"] == 16384
    )


def test_thinking_uses_env_overrides() -> bool:
    os.environ["ANTHROPIC_THINKING_BUDGET_TOKENS"] = "1234"
    os.environ["ANTHROPIC_THINKING_MAX_TOKENS"] = "5678"
    anthropic_proxy = reload_anthropic_proxy()

    normalized = anthropic_proxy.normalize_anthropic_model_and_thinking({
        "model": "claude-test-thinking",
        "messages": [{"role": "user", "content": "hello"}],
    })

    return (
        normalized["model"] == "claude-test"
        and normalized["thinking"] == {"type": "enabled", "budget_tokens": 1234}
        and normalized["max_tokens"] == 5678
    )


def test_unknown_non_thinking_model_passthrough() -> bool:
    os.environ.pop("ANTHROPIC_THINKING_BUDGET_TOKENS", None)
    os.environ.pop("ANTHROPIC_THINKING_MAX_TOKENS", None)
    anthropic_proxy = reload_anthropic_proxy()
    request_data = {
        "model": "unknown-provider-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }

    normalized = anthropic_proxy.normalize_anthropic_model_and_thinking(request_data)

    return normalized == request_data


def main() -> bool:
    tests = [
        ("cache_control 标准化不修改原始请求", test_cache_control_standardization_does_not_mutate_original),
        ("thinking 默认环境配置", test_thinking_defaults_from_env_fallback),
        ("thinking 使用环境变量覆盖", test_thinking_uses_env_overrides),
        ("未知非 thinking 模型透传", test_unknown_non_thinking_model_passthrough),
    ]

    passed = 0
    for name, test in tests:
        try:
            if test():
                print(f"✅ {name}")
                passed += 1
            else:
                print(f"❌ {name}")
        except Exception as error:
            print(f"❌ {name}: {error}")

    print(f"📊 测试结果: {passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
