#!/usr/bin/env python3
"""
Claude Proxy 综合测试套件

这个脚本包含了项目的所有测试功能：
1. Anthropic 代理测试
   - 核心端点功能测试（health, models, service）
   - 并发性能测试（最重要的测试）
2. OpenRouter 代理测试
   - 核心端点功能测试
   - 并发性能测试
   - 基本连通性测试

测试策略：
- 不测试 /v1/messages 端点，因为并发测试已经覆盖实际使用场景
- 专注于服务器基础功能和性能验证

使用方法:
    python test_suite.py
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_test_file(test_path: str, test_name: str) -> bool:
    """运行单个测试文件"""
    try:
        # 动态导入并运行测试模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            test_name,
            test_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 运行测试
        if hasattr(module, 'main'):
            # 检查是否是异步函数
            if asyncio.iscoroutinefunction(module.main):
                return asyncio.run(module.main())
            else:
                return module.main()
        else:
            print(f"⚠️  {test_name}: 无主测试函数")
            return False

    except Exception as e:
        print(f"❌ {test_name}: 异常 - {str(e)[:100]}...")
        return False

async def run_all_tests():
    """运行所有测试"""
    print("🧪 Claude Proxy 综合测试套件")
    print("=" * 80)

    # 测试配置
    tests = [
        ("Anthropic 核心端点功能", "tests/anthropic/test_anthropic_core_endpoints.py"),
        ("Anthropic 并发性能", "tests/anthropic/test_anthropic_concurrency.py"),
        ("OpenRouter 基本功能", "tests/openrouter/test_openrouter_basic.py"),
        ("OpenRouter 核心端点功能", "tests/openrouter/test_openrouter_core_endpoints.py"),
        ("OpenRouter 并发性能", "tests/openrouter/test_openrouter_concurrency.py"),
    ]

    results = {}

    for test_name, test_path in tests:
        print(f"\n🚀 运行测试: {test_name}")
        print("-" * 60)

        result = run_test_file(test_path, test_name)
        results[test_name] = "✅ 通过" if result else "❌ 失败"

    # 打印测试总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)

    for test_name, result in results.items():
        print(f"{test_name:25} : {result}")

    success_count = sum(1 for r in results.values() if r.startswith("✅"))
    total_count = len(results)

    print(f"\n🎯 总体结果: {success_count}/{total_count} 测试通过")

    if success_count == total_count:
        print("🎉 所有测试通过！系统运行正常。")
        print("\n💡 测试策略说明:")
        print("   - 核心端点测试：验证服务器基础功能")
        print("   - 并发性能测试：验证实际负载下的表现")
        print("   - 不单独测试 /v1/messages：并发测试已覆盖实际使用场景")
        print("\n🚀 你的代理系统已经准备好投入使用！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return False

def show_help():
    """显示帮助信息"""
    print("🧪 Claude Proxy 测试套件")
    print("=" * 50)
    print("可用测试:")
    print("  python test_suite.py                    # 运行所有测试")
    print("  python tests/anthropic/test_anthropic_core_endpoints.py")
    print("  python tests/anthropic/test_anthropic_concurrency.py")
    print("  python tests/openrouter/test_openrouter_basic.py")
    print("  python tests/openrouter/test_openrouter_core_endpoints.py")
    print("  python tests/openrouter/test_openrouter_concurrency.py")
    print("\n快速验证:")
    print("  1. 启动Anthropic代理: python anthropic_proxy.py")
    print("  2. 启动OpenRouter代理: python openrouter_proxy.py")
    print("  3. 运行测试: python test_suite.py")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)

    asyncio.run(run_all_tests())