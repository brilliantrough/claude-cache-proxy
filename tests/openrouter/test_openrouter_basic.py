#!/usr/bin/env python3
"""
OpenRouter API 连通性基本测试
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_openrouter_connectivity():
    """测试 OpenRouter API 基本连通性"""
    print("🧪 OpenRouter API 连通性测试")
    print("=" * 60)

    # 读取配置
    base_url = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api')
    api_key = os.getenv('OPENROUTER_API_KEY')

    # 添加 /v1 路径
    api_url = base_url + '/v1' if not base_url.endswith('/v1') else base_url

    print(f"📋 API URL: {api_url}")
    print(f"🔑 API Key: {api_key[:20]}...{api_key[-10:] if api_key and len(api_key) > 30 else 'None'}")
    print(f"📏 Key Length: {len(api_key) if api_key else 0}")

    if not api_key:
        print("❌ OPENROUTER_API_KEY 未设置")
        return False

    # 测试1：健康检查 - 获取模型列表
    print(f"\n🏥 测试1: 获取模型列表")
    try:
        models_url = f"{api_url}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"📤 GET {models_url}")
        response = requests.get(models_url, headers=headers, timeout=10)

        print(f"📊 状态码: {response.status_code}")
        print(f"📋 响应头: {dict(response.headers)}")

        # 打印原始响应内容用于调试
        print(f"📄 原始响应内容 (前200字符): {response.text[:200]}")

        if response.status_code == 200:
            try:
                data = response.json()
                model_count = len(data.get('data', []))
                print(f"✅ 模型列表获取成功，共 {model_count} 个模型")
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败: {e}")
                print(f"完整响应: {response.text}")
                return False

            # 查找目标模型
            target_model = "anthropic/claude-sonnet-4.5"
            found_models = [m for m in data.get('data', []) if target_model in m.get('id', '')]
            if found_models:
                print(f"✅ 找到目标模型: {target_model}")
            else:
                print(f"⚠️  未找到目标模型: {target_model}")
                print("💡 可用模型示例:")
                for i, model in enumerate(data.get('data', [])[:3]):
                    print(f"   - {model.get('id', 'Unknown')}")

        else:
            print(f"❌ 模型列表获取失败: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 模型列表请求异常: {e}")
        return False

    # 测试2：简单聊天请求
    print(f"\n💬 测试2: 简单聊天请求")
    try:
        chat_url = f"{api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "anthropic/claude-sonnet-4.5",
            "messages": [
                {"role": "user", "content": "Hello, please just say 'Hi there!' and nothing else."}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }

        print(f"📤 POST {chat_url}")
        print(f"📝 模型: {payload['model']}")
        print(f"📝 消息: {payload['messages'][0]['content']}")
        print(f"📝 最大tokens: {payload['max_tokens']}")

        response = requests.post(chat_url, headers=headers, json=payload, timeout=30)

        print(f"📊 状态码: {response.status_code}")
        print(f"📋 响应头: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print("✅ 聊天请求成功!")

            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice:
                    content = choice['message'].get('content', '')
                    print(f"📝 回复: '{content}'")

                if 'usage' in data:
                    usage = data['usage']
                    print(f"📊 Token使用: {usage}")

                if 'finish_reason' in choice:
                    print(f"🏁 完成原因: {choice['finish_reason']}")

            return True
        else:
            print(f"❌ 聊天请求失败")
            print(f"状态码: {response.status_code}")
            print(f"错误响应: {response.text}")

            # 分析常见错误
            if response.status_code == 401:
                print("💡 401 错误: API Key 可能无效或已过期")
            elif response.status_code == 403:
                print("💡 403 错误: 权限不足或账户问题")
            elif response.status_code == 429:
                print("💡 429 错误: 请求频率限制")
            elif response.status_code >= 500:
                print("💡 5xx 错误: OpenRouter 服务器问题")

            return False

    except Exception as e:
        print(f"❌ 聊天请求异常: {e}")
        return False

def show_next_steps(success):
    """显示后续步骤"""
    print(f"\n" + "=" * 60)
    if success:
        print("🎉 OpenRouter API 连通性测试通过!")
        print("\n📋 下一步:")
        print("1. 重启代理服务器:")
        print("   source .venv/bin/activate")
        print("   python openrouter_proxy.py")
        print("\n2. 测试代理:")
        print("   curl -X POST http://localhost:9998/v1/chat/completions \\")
        print("     -H \"Content-Type: application/json\" \\")
        print("     -d '{\"model\": \"anthropic/claude-sonnet-4.5\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 20}'")
    else:
        print("❌ OpenRouter API 连通性测试失败")
        print("\n🔧 故障排除:")
        print("1. 检查 OPENROUTER_API_KEY 是否正确")
        print("2. 检查 OpenRouter 账户余额")
        print("3. 检查网络连接")
        print("4. 访问 https://openrouter.ai 确认账户状态")

    print("=" * 60)

if __name__ == "__main__":
    success = test_openrouter_connectivity()
    show_next_steps(success)

    exit(0 if success else 1)