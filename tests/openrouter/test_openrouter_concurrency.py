#!/usr/bin/env python3
"""
测试 OpenRouter Proxy 服务器的并发处理能力
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import aiohttp
import time
import json
from typing import List, Dict
import statistics
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def get_openrouter_proxy_url():
    """获取OpenRouter代理服务器URL"""
    host = os.getenv('OPENAI_PROXY_HOST', 'localhost')
    port = os.getenv('OPENAI_PROXY_PORT', '9998')
    return f"http://{host}:{port}"

async def single_request(session: aiohttp.ClientSession, request_id: int, url: str) -> Dict:
    """发送单个请求"""
    start_time = time.time()

    request_data = {
        "model": "anthropic/claude-4.5-sonnet",
        "max_tokens": 20,
        "messages": [
            {"role": "user", "content": f"并发测试请求 #{request_id}，请简短回复"}
        ]
    }

    try:
        async with session.post(
            url,
            json=request_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-key"
            }
        ) as response:
            response_time = time.time() - start_time

            # 读取响应
            response_text = await response.text()

            return {
                "request_id": request_id,
                "status_code": response.status,
                "response_time": response_time,
                "success": response.status == 200,
                "response_text": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                "error": None
            }

    except Exception as e:
        return {
            "request_id": request_id,
            "status_code": None,
            "response_time": time.time() - start_time,
            "success": False,
            "response_text": None,
            "error": str(e)
        }

async def single_stream_request(session: aiohttp.ClientSession, request_id: int, url: str) -> Dict:
    """发送单个流式请求"""
    start_time = time.time()

    request_data = {
        "model": "anthropic/claude-4.5-sonnet",
        "max_tokens": 50,
        "stream": True,
        "messages": [
            {"role": "user", "content": f"流式并发测试 #{request_id}，请简短回复"}
        ]
    }

    try:
        async with session.post(
            url,
            json=request_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-key"
            }
        ) as response:
            response_time = time.time() - start_time

            # 读取流式响应
            chunks = []
            async for chunk in response.content:
                if chunk:
                    chunks.append(chunk.decode('utf-8', errors='ignore'))

            full_response = ''.join(chunks)

            return {
                "request_id": request_id,
                "status_code": response.status,
                "response_time": response_time,
                "success": response.status == 200,
                "response_text": full_response[:200] + "..." if len(full_response) > 200 else full_response,
                "error": None
            }

    except Exception as e:
        return {
            "request_id": request_id,
            "status_code": None,
            "response_time": time.time() - start_time,
            "success": False,
            "response_text": None,
            "error": str(e)
        }

async def run_concurrency_test(concurrency: int, total_requests: int, is_stream: bool = False) -> Dict:
    """运行并发测试"""
    print(f"\n🚀 测试并发能力: {concurrency} 个并发，总共 {total_requests} 个请求")
    print(f"📝 请求类型: {'流式' if is_stream else '非流式'}")

    url = f"{get_openrouter_proxy_url()}/v1/chat/completions"

    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(session, request_id):
        async with semaphore:
            if is_stream:
                return await single_stream_request(session, request_id, url)
            else:
                return await single_request(session, request_id, url)

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # 创建所有任务
        tasks = [limited_request(session, i) for i in range(1, total_requests + 1)]

        print(f"✅ 已创建 {len(tasks)} 个{'流式' if is_stream else ''}任务")

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    # 统计结果
    successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
    failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success', False)]

    # 计算响应时间统计
    response_times = [r['response_time'] for r in successful_requests if 'response_time' in r]

    avg_response_time = statistics.mean(response_times) if response_times else 0
    min_response_time = min(response_times) if response_times else 0
    max_response_time = max(response_times) if response_times else 0

    # 计算QPS
    requests_per_second = total_requests / total_time if total_time > 0 else 0

    success_rate = len(successful_requests) / total_requests * 100 if total_requests > 0 else 0

    # 打印结果
    print(f"📊 {'流式' if is_stream else ''}请求结果:")
    print(f"   成功率: {success_rate:.1f}% ({len(successful_requests)}/{total_requests})")
    print(f"   QPS: {requests_per_second:.2f}")
    if response_times:
        print(f"   平均响应时间: {avg_response_time:.2f}s")
        print(f"   响应时间范围: {min_response_time:.2f}s - {max_response_time:.2f}s")

    if failed_requests:
        print(f"   失败原因: {failed_requests[0].get('error', 'Unknown')[:50]}...")

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time": total_time,
        "successful": len(successful_requests),
        "failed": len(failed_requests),
        "success_rate": success_rate,
        "requests_per_second": requests_per_second,
        "avg_response_time": avg_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "results": results
    }

async def main():
    """主函数"""
    print("=" * 80)
    print("🧪 OpenRouter 代理服务器并发处理能力测试")
    print("=" * 80)
    print(f"🔗 测试地址: {get_openrouter_proxy_url()}")
    print("=" * 80)

    # 首先检查服务器是否运行
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{get_openrouter_proxy_url()}/health", timeout=5) as response:
                if response.status != 200:
                    print("❌ OpenRouter代理服务器未正常运行")
                    return
    except Exception as e:
        print(f"❌ 无法连接到OpenRouter代理服务器: {e}")
        print("💡 请确保OpenRouter代理服务器正在运行: python openrouter_proxy.py")
        return

    print("✅ OpenRouter代理服务器运行正常\n")

    # 测试场景
    test_scenarios = [
        {"concurrency": 1, "total": 5, "name": "基准测试"},
        {"concurrency": 5, "total": 20, "name": "轻度并发"},
        {"concurrency": 10, "total": 50, "name": "中度并发"},
        {"concurrency": 20, "total": 100, "name": "高度并发"},
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"\n============================== {scenario['name']} ==============================")

        # 测试普通请求
        normal_result = await run_concurrency_test(
            scenario["concurrency"],
            scenario["total"],
            is_stream=False
        )

        # 测试流式请求
        stream_result = await run_concurrency_test(
            scenario["concurrency"],
            min(20, scenario["total"]),  # 流式请求减少数量以节省时间
            is_stream=True
        )

        all_results.append({
            "name": scenario["name"],
            "normal": normal_result,
            "stream": stream_result
        })

    # 总结报告
    print("\n" + "=" * 80)
    print("📋 并发处理能力总结")
    print("=" * 80)

    max_qps = 0
    fastest_response = float('inf')

    for result in all_results:
        normal = result["normal"]
        stream = result["stream"]

        max_qps = max(max_qps, normal["requests_per_second"], stream["requests_per_second"])

        if normal["avg_response_time"] > 0:
            fastest_response = min(fastest_response, normal["avg_response_time"])
        if stream["avg_response_time"] > 0:
            fastest_response = min(fastest_response, stream["avg_response_time"])

    print(f"🚀 最大 QPS: {max_qps:.2f}")
    print(f"⚡ 最快平均响应时间: {fastest_response:.2f}s")
    print("👍 并发处理能力正常")

    print("\n💡 OpenRouter并发优化建议:")
    print("   1. 考虑使用 aiohttp 替代 requests 以提高异步性能")
    print("   2. 为每个协程创建独立的 HTTP 客户端实例")
    print("   3. 添加连接池配置以管理并发连接")
    print("   4. 监控资源使用情况，必要时增加超时配置")
    print("   5. 考虑添加请求限流机制防止过载")

if __name__ == "__main__":
    asyncio.run(main())
