#!/usr/bin/env python3
"""
测试 Claude Proxy (Anthropic) 服务器的并发处理能力
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

def get_anthropic_proxy_url():
    """获取Anthropic代理服务器URL"""
    host = os.getenv('PROXY_HOST', 'localhost')
    port = os.getenv('PROXY_PORT', '8080')
    return f"http://{host}:{port}"

async def single_request(session: aiohttp.ClientSession, request_id: int, url: str) -> Dict:
    """发送单个请求"""
    start_time = time.time()

    request_data = {
        "model": "claude-sonnet-4-5-20250929",
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
                "x-api-key": "test-key"
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:

            elapsed_time = time.time() - start_time

            if response.status == 200:
                result = await response.json()
                return {
                    "request_id": request_id,
                    "success": True,
                    "status_code": response.status,
                    "response_time": elapsed_time,
                    "response_length": len(json.dumps(result)),
                    "error": None
                }
            else:
                error_text = await response.text()
                return {
                    "request_id": request_id,
                    "success": False,
                    "status_code": response.status,
                    "response_time": elapsed_time,
                    "response_length": len(error_text),
                    "error": f"HTTP {response.status}: {error_text[:100]}"
                }

    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "status_code": None,
            "response_time": elapsed_time,
            "response_length": 0,
            "error": "请求超时"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "status_code": None,
            "response_time": elapsed_time,
            "response_length": 0,
            "error": str(e)
        }

async def test_concurrent_requests(concurrency: int = 10, total_requests: int = 50) -> Dict:
    """测试并发请求"""
    print(f"🚀 测试并发能力: {concurrency} 个并发，总共 {total_requests} 个请求")

    url = f"{get_anthropic_proxy_url()}/v1/messages"

    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session: aiohttp.ClientSession, request_id: int) -> Dict:
        async with semaphore:
            return await single_request(session, request_id, url)

    start_time = time.time()

    # 使用 aiohttp 替代 requests 的异步客户端
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 创建所有任务
        tasks = [
            bounded_request(session, i)
            for i in range(1, total_requests + 1)
        ]

        print(f"✅ 已创建 {len(tasks)} 个并发任务")

        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    # 统计结果
    successful_requests = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success')]
    exceptions = [r for r in results if not isinstance(r, dict)]

    # 计算性能指标
    if successful_requests:
        response_times = [r['response_time'] for r in successful_requests]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        median_response_time = statistics.median(response_times)
    else:
        avg_response_time = min_response_time = max_response_time = median_response_time = 0

    success_rate = len(successful_requests) / total_requests * 100
    requests_per_second = total_requests / total_time if total_time > 0 else 0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time": total_time,
        "successful": len(successful_requests),
        "failed": len(failed_requests),
        "exceptions": len(exceptions),
        "success_rate": success_rate,
        "requests_per_second": requests_per_second,
        "avg_response_time": avg_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "median_response_time": median_response_time,
        "results": results
    }

async def test_streaming_concurrency(concurrency: int = 5, total_requests: int = 20) -> Dict:
    """测试流式并发请求"""
    print(f"🌊 测试流式并发: {concurrency} 个并发，总共 {total_requests} 个请求")

    url = f"{get_anthropic_proxy_url()}/v1/messages"

    async def stream_request(session: aiohttp.ClientSession, request_id: int) -> Dict:
        start_time = time.time()

        request_data = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 30,
            "stream": True,
            "messages": [
                {"role": "user", "content": f"流式并发测试 #{request_id}"}
            ]
        }

        try:
            async with session.post(
                url,
                json=request_data,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": "test-key"
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:

                content_chunks = []
                async for chunk in response.content:
                    if chunk:
                        content_chunks.append(chunk)

                elapsed_time = time.time() - start_time
                total_content = b''.join(content_chunks)

                return {
                    "request_id": request_id,
                    "success": response.status == 200,
                    "status_code": response.status,
                    "response_time": elapsed_time,
                    "chunks_received": len(content_chunks),
                    "content_length": len(total_content),
                    "error": None
                }

        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "status_code": None,
                "response_time": elapsed_time,
                "chunks_received": 0,
                "content_length": 0,
                "error": str(e)
            }

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_stream_request(session: aiohttp.ClientSession, request_id: int) -> Dict:
        async with semaphore:
            return await stream_request(session, request_id)

    start_time = time.time()

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            bounded_stream_request(session, i)
            for i in range(1, total_requests + 1)
        ]

        print(f"✅ 已创建 {len(tasks)} 个流式任务")
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    successful_requests = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success')]

    success_rate = len(successful_requests) / total_requests * 100
    requests_per_second = total_requests / total_time if total_time > 0 else 0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time": total_time,
        "successful": len(successful_requests),
        "failed": len(failed_requests),
        "success_rate": success_rate,
        "requests_per_second": requests_per_second,
        "results": results
    }

async def main():
    """主函数"""
    print("=" * 80)
    print("🧪 代理服务器并发处理能力测试")
    print("=" * 80)
    print(f"🔗 测试地址: {get_anthropic_proxy_url()}")
    print("=" * 80)

    # 首先检查服务器是否运行
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{get_anthropic_proxy_url()}/health", timeout=5) as response:
                if response.status != 200:
                    print("❌ 代理服务器未正常运行")
                    return
    except Exception as e:
        print(f"❌ 无法连接到代理服务器: {e}")
        print("💡 请确保代理服务器正在运行: python anthropic_proxy.py")
        return

    print("✅ 代理服务器运行正常\n")

    # 测试场景
    test_scenarios = [
        {"concurrency": 1, "total": 5, "name": "基准测试"},
        {"concurrency": 5, "total": 20, "name": "轻度并发"},
        {"concurrency": 10, "total": 50, "name": "中度并发"},
        {"concurrency": 20, "total": 100, "name": "高度并发"},
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"\n{'='*30} {scenario['name']} {'='*30}")

        # 测试普通请求
        result = await test_concurrent_requests(
            concurrency=scenario['concurrency'],
            total_requests=scenario['total']
        )
        result['test_type'] = 'normal'
        result['scenario_name'] = scenario['name']
        all_results.append(result)

        print(f"📊 普通请求结果:")
        print(f"   成功率: {result['success_rate']:.1f}% ({result['successful']}/{result['total_requests']})")
        print(f"   QPS: {result['requests_per_second']:.2f}")
        print(f"   平均响应时间: {result['avg_response_time']:.2f}s")
        print(f"   响应时间范围: {result['min_response_time']:.2f}s - {result['max_response_time']:.2f}s")

        # 测试流式请求
        if scenario['concurrency'] <= 10:  # 流式测试降低并发
            stream_result = await test_streaming_concurrency(
                concurrency=scenario['concurrency'],
                total_requests=min(scenario['total'], 20)
            )
            stream_result['test_type'] = 'stream'
            stream_result['scenario_name'] = scenario['name']
            all_results.append(stream_result)

            print(f"📊 流式请求结果:")
            print(f"   成功率: {stream_result['success_rate']:.1f}% ({stream_result['successful']}/{stream_result['total_requests']})")
            print(f"   QPS: {stream_result['requests_per_second']:.2f}")

    # 总结
    print(f"\n{'='*80}")
    print("📋 并发处理能力总结")
    print(f"{'='*80}")

    normal_results = [r for r in all_results if r['test_type'] == 'normal']
    if normal_results:
        max_qps = max(r['requests_per_second'] for r in normal_results)
        min_response_time = min(r['avg_response_time'] for r in normal_results)

        print(f"🚀 最大 QPS: {max_qps:.2f}")
        print(f"⚡ 最快平均响应时间: {min_response_time:.2f}s")

        # 分析并发性能
        baseline = normal_results[0]  # 第一个测试作为基准
        high_concurrency = normal_results[-1]  # 最后一个测试作为高并发

        if baseline['success_rate'] >= 90 and high_concurrency['success_rate'] >= 90:
            print("✅ 并发处理能力良好，高并发下成功率保持稳定")
        elif high_concurrency['success_rate'] < baseline['success_rate'] - 10:
            print("⚠️  高并发下性能下降明显，可能存在并发瓶颈")
        else:
            print("👍 并发处理能力正常")

    print("\n💡 并发优化建议:")
    print("   1. 考虑使用 aiohttp 替代 requests 以提高异步性能")
    print("   2. 为每个协程创建独立的 HTTP 客户端实例")
    print("   3. 添加连接池配置以管理并发连接")
    print("   4. 监控资源使用情况，必要时增加超时配置")
    print("   5. 考虑添加请求限流机制防止过载")

if __name__ == "__main__":
    asyncio.run(main())
