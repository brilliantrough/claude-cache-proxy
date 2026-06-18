#!/usr/bin/env python3
"""
Claude Proxy (Anthropic) 上游错误处理测试

复现并验证 bug：当上游 API 返回非 2xx 状态码时，
aiohttp 会抛出 ClientResponseError，而该异常对象 **没有** .text()/.json() 方法。
旧代码在 except 分支里调用 `await e.text()`，会再次抛出 AttributeError，
导致真实的上游错误被掩盖成笼统的 500 internal_error。

本测试通过 mock aiohttp 会话来模拟上游返回 400 错误，
不需要真实网络或 API Key。
"""

import sys
import os
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
from yarl import URL

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from anthropic_proxy import AnthropicRequestHandler


UPSTREAM_ERROR_BODY = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "model: claude-opus-4-7 is not a valid model name",
    },
}


def _make_response_cm(*, status, body_text, content_type="application/json"):
    """构造一个模拟 aiohttp 响应的 async context manager。

    raise_for_status() 会像真实 aiohttp 一样抛出 ClientResponseError，
    且该异常对象不携带 body（与真实行为一致）。
    """
    response = MagicMock()
    response.status = status
    response.headers = {"content-type": content_type}

    async def _text():
        return body_text

    async def _json():
        import json as _json_mod
        return _json_mod.loads(body_text)

    response.text = _text
    response.json = _json

    def _raise_for_status():
        request_info = MagicMock()
        request_info.real_url = URL("https://aihubmix.com/v1/messages")
        raise aiohttp.ClientResponseError(
            request_info,
            (),
            status=status,
            message="Bad Request",
            headers={"content-type": content_type},
        )

    response.raise_for_status = _raise_for_status

    class _CM:
        async def __aenter__(self_inner):
            return response

        async def __aexit__(self_inner, exc_type, exc, tb):
            return False

    return _CM()


def _make_handler_with_error(status, body_text):
    handler = AnthropicRequestHandler("https://aihubmix.com", "test-key", "5m")

    fake_session = MagicMock()
    fake_session.post = MagicMock(
        return_value=_make_response_cm(status=status, body_text=body_text)
    )

    async def _fake_get_session():
        return fake_session

    handler._get_session = _fake_get_session
    return handler


def test_non_stream_upstream_error_is_surfaced():
    """非流式：上游 400 错误应被如实返回，而不是变成 500 internal_error。"""
    import json
    body = json.dumps(UPSTREAM_ERROR_BODY)
    handler = _make_handler_with_error(400, body)

    result = asyncio.run(handler._forward_to_anthropic({"model": "x", "messages": []}))

    print(f"  非流式返回: {result}")

    # 不应是 AttributeError 掩盖后的 internal_error
    err_type = result.get("error", {}).get("type")
    err_msg = result.get("error", {}).get("message", "")

    assert "has no attribute" not in err_msg, (
        f"❌ 仍触发 AttributeError，错误处理器自身崩溃: {err_msg}"
    )
    assert err_type != "internal_error", (
        f"❌ 上游 400 被错误地归类为 internal_error: {result}"
    )

    # 理想情况下，应能拿到上游真实的错误体
    assert result.get("error", {}).get("type") == "invalid_request_error", (
        f"❌ 未能透传上游真实错误体: {result}"
    )
    print("  ✅ 非流式上游错误已正确透传")


def test_get_request_upstream_error_is_surfaced():
    """GET（如 models 端点）：上游错误也不应触发 AttributeError。"""
    import json
    body = json.dumps(UPSTREAM_ERROR_BODY)
    handler = AnthropicRequestHandler("https://aihubmix.com", "test-key", "5m")

    fake_session = MagicMock()
    fake_session.get = MagicMock(
        return_value=_make_response_cm(status=401, body_text=body)
    )

    async def _fake_get_session():
        return fake_session

    handler._get_session = _fake_get_session

    result = asyncio.run(handler.handle_get_request("/v1/models"))
    print(f"  GET 返回: {result}")

    err_msg = result.get("error", {}).get("message", "")
    assert "has no attribute" not in err_msg, (
        f"❌ GET 分支仍触发 AttributeError: {err_msg}"
    )
    print("  ✅ GET 上游错误已正确处理")


def main():
    print("🧪 测试 Anthropic 代理上游错误处理...")
    tests = [
        ("非流式上游错误透传", test_non_stream_upstream_error_is_surfaced),
        ("GET 上游错误透传", test_get_request_upstream_error_is_surfaced),
    ]
    passed = 0
    for name, fn in tests:
        print(f"\n▶ {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  {e}")
        except Exception as e:
            print(f"  ❌ 测试异常: {type(e).__name__}: {e}")

    print(f"\n{'='*40}")
    print(f"通过 {passed}/{len(tests)}")
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
