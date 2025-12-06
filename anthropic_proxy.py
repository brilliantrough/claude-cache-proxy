import os
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import aiohttp
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置，覆盖现有配置
)
logger = logging.getLogger(__name__)


class AnthropicRequestHandler:
    """Anthropic API 请求处理器，负责转发请求"""

    def __init__(self, api_url: str, api_key: str, cache_control_ttl: str = "1h"):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.cache_control_ttl = cache_control_ttl
        self.session: Optional[aiohttp.ClientSession] = None

        # 检查API URL是否已经包含完整路径
        if self.api_url.endswith('/v1/messages'):
            self.messages_endpoint = self.api_url
        else:
            self.messages_endpoint = f"{self.api_url}/v1/messages"

        logger.info(f"Initialized with cache_control TTL: {cache_control_ttl}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp 会话"""
        if self.session is None or self.session.closed:
            # 优化超时配置以支持长上下文和并发
            timeout = aiohttp.ClientTimeout(
                total=300,  # 总超时增加到5分钟
                connect=60,  # 连接超时1分钟
                sock_read=240  # 读取超时4分钟
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=50,  # 减少总连接池大小避免资源竞争
                    limit_per_host=10,  # 减少每个主机的连接数
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    # 启用keep-alive连接
                    keepalive_timeout=30,
                    # 启用TCP缓冲区自动调整
                    enable_cleanup_closed=True,
                )
            )
        return self.session

    def _prepare_headers(self, original_headers: Dict[str, str]) -> Dict[str, str]:
        """准备请求头，使用配置中的默认API Key，不转发客户端头部"""
        headers = {}

        # 只添加必要的头部，不转发客户端的敏感头部
        headers['x-api-key'] = self.api_key
        headers['anthropic-version'] = original_headers.get('anthropic-version', '2023-06-01')
        headers['content-type'] = 'application/json'

        # 保留一些可能需要的客户端头部（可选）
        optional_headers = ['anthropic-beta', 'user-agent']
        for header in optional_headers:
            if header in original_headers:
                headers[header] = original_headers[header]

        return headers

    def _standardize_cache_control(self, messages: list) -> list:
        """标准化所有消息的 cache_control 策略
        - 清理所有消息的 cache_control
        - 只在最后一条消息添加标准的 cache_control
        """
        logger.info(f"{messages}")
        if not messages:
            return messages

        standardized_messages = []

        # 处理所有消息的 cache_control
        for i, message in enumerate(messages):
            cleaned_message = message.copy()

            # 清理 message 级别的 cache_control
            if 'cache_control' in cleaned_message:
                del cleaned_message['cache_control']

            # 清理 content 级别的 cache_control
            content = message.get('content', [])
            if isinstance(content, list):
                cleaned_content = []
                for content_block in content:
                    if isinstance(content_block, dict):
                        # 移除 cache_control 字段，保留其他字段
                        cleaned_block = {k: v for k, v in content_block.items() if k != 'cache_control'}
                        cleaned_content.append(cleaned_block)
                    else:
                        cleaned_content.append(content_block)
                cleaned_message['content'] = cleaned_content
            # 字符串内容保持不变（不包含 cache_control）

            standardized_messages.append(cleaned_message)

        # 在最后一条消息添加标准的 cache_control
        if standardized_messages:
            last_message = standardized_messages[-1]
            content = last_message.get('content', [])

            if isinstance(content, list):
                # 在最后一个 content 块中添加 cache_control
                if content:
                    # 如果有 content 块，在最后一个块中添加 cache_control
                    if isinstance(content[-1], dict):
                        content[-1]['cache_control'] = {"type": "ephemeral", "ttl": self.cache_control_ttl}
                    else:
                        # 如果最后一个块不是 dict，创建新的 dict 块
                        content.append({
                            "type": "text",
                            "text": str(content[-1]) if content[-1] else "",
                            "cache_control": {"type": "ephemeral", "ttl": self.cache_control_ttl}
                        })
                else:
                    # 如果 content 为空，创建新的 content 块
                    content = [{
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral", "ttl": self.cache_control_ttl}
                    }]
            elif isinstance(content, str):
                # 将字符串转换为 list 格式并添加 cache_control
                last_message['content'] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral", "ttl": self.cache_control_ttl}
                    }
                ]
            else:
                # 如果 content 为 None 或其他格式，创建新的 content
                last_message['content'] = [
                    {
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral", "ttl": self.cache_control_ttl}
                    }
                ]

        logger.info(f"Standardized cache_control: cleared all cache_control, added TTL={self.cache_control_ttl} to last message")
        logger.info(standardized_messages)
        return standardized_messages

    def _has_cache_control(self, messages: list) -> bool:
        """检查消息是否包含缓存控制"""
        for message in messages:
            # 检查 message 级别的 cache_control
            if 'cache_control' in message:
                return True

            # 检查 content 级别的 cache_control
            content = message.get('content', [])
            if isinstance(content, list):
                for content_block in content:
                    if isinstance(content_block, dict) and 'cache_control' in content_block:
                        return True
            elif isinstance(content, str):
                # 简单字符串内容，不支持缓存控制
                pass
        return False

    def _validate_anthropic_request(self, request_data: Dict[str, Any]) -> bool:
        """验证Anthropic请求格式"""
        required_fields = ['model', 'messages']

        for field in required_fields:
            if field not in request_data:
                logger.error(f"Missing required field: {field}")
                return False

        messages = request_data.get('messages', [])
        if not isinstance(messages, list) or len(messages) == 0:
            logger.error("Messages must be a non-empty list")
            return False

        for message in messages:
            if not isinstance(message, dict):
                logger.error("Each message must be a dictionary")
                return False
            if 'role' not in message or 'content' not in message:
                logger.error("Each message must have 'role' and 'content' fields")
                return False

        return True

    async def handle_request(self, request_data: Dict[str, Any],
                           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """处理Anthropic API请求"""
        headers = headers or {}

        # 验证请求格式
        if not self._validate_anthropic_request(request_data):
            raise ValueError("Invalid Anthropic request format")

        messages = request_data.get('messages', [])

        # 标准化缓存策略：清理所有 cache_control，只在最后一条添加标准配置
        standardized_messages = self._standardize_cache_control(messages)

        # 更新请求中的消息
        standardized_request = request_data.copy()
        standardized_request['messages'] = standardized_messages

        logger.info("Request processed with standardized cache_control strategy")

        # 直接转发请求给 Anthropic API
        response_data = await self._forward_to_anthropic(standardized_request, headers)
        return response_data

    async def handle_stream_request(self, request_data: Dict[str, Any],
                                  headers: Optional[Dict[str, str]] = None) -> AsyncGenerator[bytes, None]:
        """处理流式 Anthropic API 请求"""
        headers = headers or {}

        # 验证请求格式
        if not self._validate_anthropic_request(request_data):
            raise ValueError("Invalid Anthropic request format")

        messages = request_data.get('messages', [])

        # 标准化缓存策略：清理所有 cache_control，只在最后一条添加标准配置
        standardized_messages = self._standardize_cache_control(messages)

        # 更新请求中的消息
        standardized_request = request_data.copy()
        standardized_request['messages'] = standardized_messages

        logger.info("Stream request processed with standardized cache_control strategy")

        # 转发流式请求给 Anthropic API
        async for chunk in self._forward_stream_to_anthropic(standardized_request, headers):
            yield chunk

    async def handle_get_request(self, endpoint: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """处理GET请求（如models端点）"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()

        # 构建完整的URL
        if endpoint.startswith('/'):
            url = f"{self.api_url.rstrip('/')}{endpoint}"
        else:
            url = f"{self.api_url}/{endpoint}"

        logger.info(f"Forwarding GET request to: {url}")

        try:
            async with session.get(url, headers=prepared_headers) as response:
                if response.headers.get('content-type', '').startswith('application/json'):
                    return await response.json()
                else:
                    return {"data": await response.text()}

        except asyncio.TimeoutError:
            logger.error(f"Timeout error from Anthropic API: request timed out after 120 seconds")
            return {
                "error": {
                    "type": "timeout_error",
                    "message": "Request to Anthropic API timed out after 120 seconds"
                }
            }

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error from Anthropic API: {e.status} - {await e.text()}")
            error_response = {
                "error": {
                    "type": "api_error",
                    "message": f"Anthropic API returned status {e.status}"
                }
            }
            if e.headers.get('content-type', '').startswith('application/json'):
                try:
                    error_response = await e.json()
                except:
                    pass
            return error_response

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            return {
                "error": {
                    "type": "network_error",
                    "message": f"Network error: {str(e)}"
                }
            }

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "error": {
                    "type": "internal_error",
                    "message": f"Internal server error: {str(e)}"
                }
            }

    async def _forward_stream_to_anthropic(self, request_data: Dict[str, Any],
                                      headers: Optional[Dict[str, str]] = None) -> AsyncGenerator[bytes, None]:
        """转发流式请求到Anthropic API"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()
        url = self.messages_endpoint

        logger.info(f"Forwarding stream request to: {url}")
        logger.info(f"Headers: {prepared_headers}")
        logger.info(f"Request data: {request_data}")


        try:
            async with session.post(
                url,
                json=request_data,
                headers=prepared_headers,
                # 优化流式响应配置
                read_bufsize=8192,
                # 禁用自动解压缩，处理原始流
                auto_decompress=False,
            ) as response:
                response.raise_for_status()

                logger.info(f"Successfully connected to streaming endpoint")

                # 使用更健壮的流式读取方法
                try:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        yield chunk
                except (aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
                    logger.error(f"Streaming interrupted: {str(e)}")
                    # 发送SSE格式的错误信息
                    error_data = {
                        "error": {
                            "type": "stream_interrupted",
                            "message": f"Streaming interrupted due to: {str(e)}"
                        }
                    }
                    error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
                    yield error_chunk
                    return

                logger.info(f"Streaming request completed successfully")

        except asyncio.TimeoutError:
            logger.error(f"Timeout error from Anthropic API: stream request timed out")
            error_data = {
                "error": {
                    "type": "timeout_error",
                    "message": "Request to Anthropic API timed out"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error from Anthropic API: {e.status} - {str(e)}")
            error_data = {
                "error": {
                    "type": "api_error",
                    "message": f"Anthropic API returned status {e.status}: {str(e)}"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            error_data = {
                "error": {
                    "type": "network_error",
                    "message": f"Network error: {str(e)}"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except Exception as e:
            logger.error(f"Unexpected error during streaming: {str(e)}")
            error_data = {
                "error": {
                    "type": "unknown_error",
                    "message": f"Unexpected error: {str(e)}"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

    async def _forward_to_anthropic(self, request_data: Dict[str, Any],
                                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """转发请求到Anthropic API"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()
        url = self.messages_endpoint

        logger.info(f"Forwarding request to: {url}")

        try:
            async with session.post(
                url,
                json=request_data,
                headers=prepared_headers
            ) as response:
                response.raise_for_status()

                response_data = await response.json()
                logger.info(f"Successfully received response from Anthropic API")
                return response_data

        except asyncio.TimeoutError:
            logger.error(f"Timeout error from Anthropic API: request timed out after 120 seconds")
            return {
                "error": {
                    "type": "timeout_error",
                    "message": "Request to Anthropic API timed out after 120 seconds"
                }
            }

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error from Anthropic API: {e.status} - {await e.text()}")
            error_response = {
                "error": {
                    "type": "api_error",
                    "message": f"Anthropic API returned status {e.status}"
                }
            }
            if e.headers.get('content-type', '').startswith('application/json'):
                try:
                    error_response = await e.json()
                except:
                    pass
            return error_response

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            return {
                "error": {
                    "type": "network_error",
                    "message": f"Network error: {str(e)}"
                }
            }

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "error": {
                    "type": "internal_error",
                    "message": f"Internal server error: {str(e)}"
                }
            }

    async def close(self):
        """清理资源"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("HTTP session closed")

    def __del__(self):
        """清理资源"""
        # 注意：__del 中不能使用 await，这里只是作为备用
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                # 创建任务来关闭会话
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    asyncio.create_task(self.close())
            except:
                pass


# 创建FastAPI应用
app = FastAPI(
    title="Claude Proxy",
    description="A proxy server for Claude API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局请求处理器
request_handler: Optional[AnthropicRequestHandler] = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global request_handler

    # 读取配置
    api_url = os.getenv('ANTHROPIC_API_URL', 'https://api.anthropic.com')
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # 读取缓存控制配置
    cache_control_ttl = os.getenv('CACHE_CONTROL_TTL', '1h')

    # 验证缓存控制配置
    if cache_control_ttl not in ['5m', '1h']:
        logger.warning(f"Invalid CACHE_CONTROL_TTL '{cache_control_ttl}', using default '1h'")
        cache_control_ttl = '1h'

    # 初始化请求处理器
    request_handler = AnthropicRequestHandler(api_url, api_key, cache_control_ttl)

    logger.info(f"Claude Proxy server started")
    logger.info(f"Target API URL: {api_url}")
    logger.info(f"Cache Control TTL: {cache_control_ttl}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    if request_handler:
        await request_handler.close()
    logger.info("Claude Proxy server shutdown")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "Claude Proxy",
        "status": "running",
        "endpoints": {
            "/": "Service information",
            "/v1/messages": "Anthropic compatible messages endpoint (POST)",
            "/v1/models": "Anthropic compatible models endpoint (GET)",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "claude-proxy"}


@app.get("/v1/models")
async def models_endpoint(request: Request):
    """Anthropic 兼容的模型端点"""
    if not request_handler:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # 获取请求头
        headers = dict(request.headers)

        # 转发GET请求到Anthropic API
        logger.info("Received models endpoint request")
        response_data = await request_handler.handle_get_request("/v1/models", headers)

        # 检查是否是错误响应
        if 'error' in response_data:
            error_type = response_data['error'].get('type', 'unknown_error')
            status_code = 500

            if error_type == 'api_error':
                status_code = 400
            elif error_type == 'network_error':
                status_code = 503

            logger.warning(f"Models request failed: {response_data}")
            return JSONResponse(status_code=status_code, content=response_data)

        logger.info("Models request completed successfully")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Unexpected error in models endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/messages")
async def messages_endpoint(request: Request):
    """Anthropic 兼容的消息端点，支持流式和非流式回复"""
    if not request_handler:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # 获取请求内容
        content_type = request.headers.get('content-type', '')
        if not content_type.startswith('application/json'):
            raise HTTPException(status_code=400, detail="Content-Type must be application/json")

        request_data = await request.json()

        # 检查是否为流式请求
        is_stream = request_data.get('stream', False)
        request_data.pop("top_p")

        # 获取请求头
        headers = dict(request.headers)

        # 处理请求
        logger.info(f"Received {'stream' if is_stream else 'non-stream'} request")
        logger.info(f"Request data: {json.dumps({k: v for k, v in request_data.items() if k != 'messages'}, ensure_ascii=False)}")

        if is_stream:
            # 流式回复
            logger.info("Processing as streaming request")
            return StreamingResponse(
                request_handler.handle_stream_request(request_data, headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        else:
            # 非流式回复
            logger.info("Processing as non-streaming request")
            response_data = await request_handler.handle_request(request_data, headers)

            # 检查是否是错误响应
            if 'error' in response_data:
                error_type = response_data['error'].get('type', 'unknown_error')
                status_code = 500

                if error_type == 'api_error':
                    status_code = 400
                elif error_type == 'network_error':
                    status_code = 503
                elif error_type == 'timeout_error':
                    status_code = 408
                elif error_type == 'json_decode_error':
                    status_code = 502

                logger.warning(f"Request failed: {response_data}")
                return JSONResponse(status_code=status_code, content=response_data)

            logger.info("Request completed successfully")
            return JSONResponse(content=response_data)

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    # 读取服务器配置
    host = os.getenv('PROXY_HOST', '0.0.0.0')
    port = int(os.getenv('PROXY_PORT', 8080))

    logger.info(f"Starting Claude Proxy server on {host}:{port}")

    # 启动服务器
    uvicorn.run(
        "anthropic_proxy:app",
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )
