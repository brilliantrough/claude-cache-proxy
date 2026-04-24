#!/usr/bin/env python3
"""OpenRouter 模型映射加载测试。"""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def reload_openrouter_proxy(config_path: Path, env_mapping=None):
    os.environ["OPENROUTER_MODEL_MAP_FILE"] = str(config_path)
    if env_mapping is None:
        os.environ.pop("OPENROUTER_MODEL_MAP_JSON", None)
    else:
        os.environ["OPENROUTER_MODEL_MAP_JSON"] = json.dumps(env_mapping)

    import openrouter_proxy

    return importlib.reload(openrouter_proxy)


def test_load_model_map_from_json_file() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "openrouter_model_map.json"
        config_path.write_text(
            json.dumps({"claude-opus-4-7": "anthropic/claude-opus-4.7"}),
            encoding="utf-8",
        )
        openrouter_proxy = reload_openrouter_proxy(config_path)

        model_map = openrouter_proxy.load_model_map_from_env()
        return model_map["claude-opus-4-7"] == "anthropic/claude-opus-4.7"


def test_env_mapping_overrides_file_mapping() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "openrouter_model_map.json"
        config_path.write_text(
            json.dumps({"claude-opus-latest": "anthropic/claude-opus-4.7"}),
            encoding="utf-8",
        )
        openrouter_proxy = reload_openrouter_proxy(
            config_path,
            env_mapping={"claude-opus-latest": "anthropic/claude-opus-override"},
        )

        model_map = openrouter_proxy.load_model_map_from_env()
        return model_map["claude-opus-latest"] == "anthropic/claude-opus-override"


def test_thinking_alias_uses_file_mapping() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "openrouter_model_map.json"
        config_path.write_text(
            json.dumps({"claude-opus-4-7": "anthropic/claude-opus-4.7"}),
            encoding="utf-8",
        )
        openrouter_proxy = reload_openrouter_proxy(config_path)

        handler = openrouter_proxy.OpenAIRequestHandler("https://openrouter.ai/api", "test-key")
        normalized = handler._normalize_model_and_reasoning({
            "model": "claude-opus-4-7-thinking",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        return (
            normalized["original_model"] == "claude-opus-4-7-thinking"
            and normalized["model"] == "anthropic/claude-opus-4.7"
            and normalized["thinking"] == {"type": "enabled", "budget_tokens": 30000}
            and normalized["max_tokens"] == 80000
        )


def main() -> bool:
    tests = [
        ("JSON 文件映射加载", test_load_model_map_from_json_file),
        ("环境变量覆盖 JSON 文件", test_env_mapping_overrides_file_mapping),
        ("thinking 别名使用 JSON 映射", test_thinking_alias_uses_file_mapping),
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
