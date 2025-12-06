#!/usr/bin/env python3
"""
Claude Proxy (Anthropic) 核心功能测试

专注于核心端点的基本功能验证：
1. 健康检查
2. 模型列表获取
3. 服务信息

不测试 /v1/messages，因为并发测试已经覆盖了实际使用场景
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def get_anthropic_proxy_url():
    """获取Anthropic代理服务器URL"""
    host = os.getenv('PROXY_HOST', 'localhost')
    port = os.getenv('PROXY_PORT', '8080')
    return f"http://{host}:{port}"

def test_health_endpoint():
    """测试健康检查端点"""
    print("🏥 测试 /health 端点...")
    try:
        base_url = get_anthropic_proxy_url()
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ /health 端点正常: {data}")
            return True
        else:
            print(f"❌ /health 端点失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ /health 端点异常: {e}")
        print(f"💡 请确保代理服务器正在运行: python anthropic_proxy.py")
        return False

def test_models_endpoint():
    """测试 models 端点"""
    print("\n🤖 测试 /v1/models 端点...")
    try:
        base_url = get_anthropic_proxy_url()
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and isinstance(data['data'], list):
                print(f"✅ /v1/models 端点正常，返回 {len(data['data'])} 个模型")
                # 显示前几个模型
                for i, model in enumerate(data['data'][:3]):
                    print(f"   - {model.get('id', 'Unknown')}")
                return True
            else:
                print(f"❌ /v1/models 返回格式错误: {data}")
                return False
        else:
            print(f"❌ /v1/models 端点失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ /v1/models 端点异常: {e}")
        print(f"💡 请确保代理服务器正在运行: python anthropic_proxy.py")
        return False

def test_service_info():
    """测试服务信息端点"""
    print("\n📋 测试服务信息端点...")
    try:
        base_url = get_anthropic_proxy_url()
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务信息正常: {data.get('service')} - {data.get('status')}")
            print(f"   可用端点: {list(data.get('endpoints', {}).keys())}")
            return True
        else:
            print(f"❌ 服务信息端点失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 服务信息端点异常: {e}")
        print(f"💡 请确保代理服务器正在运行: python anthropic_proxy.py")
        return False

def main():
    """运行核心功能测试"""
    print("🧪 Claude Proxy 核心功能测试")
    print("=" * 50)
    print(f"🔗 测试地址: {get_anthropic_proxy_url()}")
    print("=" * 50)

    tests = [
        test_health_endpoint,
        test_models_endpoint,
        test_service_info,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # 短暂间隔

    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有核心端点测试通过！")
        print("💡 接下来运行并发测试验证实际性能:")
        print("   python test_concurrency.py")
        return True
    else:
        print("⚠️  部分测试失败，请检查服务状态")
        print("💡 启动服务器命令:")
        print("   source .venv/bin/activate")
        print("   python anthropic_proxy.py")
        return False

if __name__ == "__main__":
    main()