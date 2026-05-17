import os
import json
import logging
import asyncio
import uuid
from typing import Optional, AsyncGenerator, Dict, Any
from fastapi import FastAPI, Request, HTTPException
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


def _summarize_content_block(content: Any) -> Dict[str, Any]:
    """生成消息内容摘要，避免日志打印正文或 base64 数据"""
    if isinstance(content, str):
        return {
            "content_type": "text",
            "text_length": len(content)
        }

    if isinstance(content, list):
        block_types = []
        image_count = 0
        text_blocks = 0
        tool_use_blocks = 0
        tool_result_blocks = 0

        for block in content:
            if not isinstance(block, dict):
                block_types.append(type(block).__name__)
                continue

            block_type = block.get('type', 'unknown')
            block_types.append(block_type)

            if block_type == 'image':
                image_count += 1
            elif block_type == 'text':
                text_blocks += 1
            elif block_type == 'tool_use':
                tool_use_blocks += 1
            elif block_type == 'tool_result':
                tool_result_blocks += 1

        return {
            "content_type": "blocks",
            "block_count": len(content),
            "block_types": block_types,
            "image_count": image_count,
            "text_blocks": text_blocks,
            "tool_use_blocks": tool_use_blocks,
            "tool_result_blocks": tool_result_blocks,
        }

    return {
        "content_type": type(content).__name__
    }


def summarize_request_for_logging(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成安全的请求摘要，避免日志输出完整消息内容"""
    summary = {
        "model": request_data.get("model"),
        "stream": request_data.get("stream", False),
        "max_tokens": request_data.get("max_tokens"),
        "temperature": request_data.get("temperature"),
        "top_p": request_data.get("top_p"),
        "tools_count": len(request_data.get("tools", [])) if isinstance(request_data.get("tools"), list) else None,
        "system_count": len(request_data.get("system", [])) if isinstance(request_data.get("system"), list) else (1 if request_data.get("system") else 0),
    }

    messages = request_data.get("messages", [])
    if isinstance(messages, list):
        summary["messages_count"] = len(messages)
        summary["message_roles"] = [message.get("role", "unknown") for message in messages if isinstance(message, dict)]

        if messages and isinstance(messages[-1], dict):
            summary["last_message"] = {
                "role": messages[-1].get("role", "unknown"),
                **_summarize_content_block(messages[-1].get("content"))
            }

    thinking = request_data.get("thinking")
    if isinstance(thinking, dict):
        summary["thinking"] = {
            "type": thinking.get("type"),
            "budget_tokens": thinking.get("budget_tokens")
        }

    return summary


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        logger.warning(f"Invalid {name} '{raw_value}', using default {default}")
        return default


def normalize_anthropic_model_and_thinking(request_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized_request = request_data.copy()
    model_name = normalized_request.get('model', '')

    if not isinstance(model_name, str) or not model_name.endswith('-thinking'):
        return normalized_request

    base_model = model_name[:-len('-thinking')]
    thinking_budget = _read_int_env('ANTHROPIC_THINKING_BUDGET_TOKENS', 8192)
    thinking_max_tokens = _read_int_env('ANTHROPIC_THINKING_MAX_TOKENS', 16384)

    normalized_request['model'] = base_model
    normalized_request['thinking'] = {
        "type": "enabled",
        "budget_tokens": thinking_budget
    }
    normalized_request['max_tokens'] = thinking_max_tokens

    logger.info(f"Processed thinking model: {model_name} -> {base_model} with thinking enabled")
    return normalized_request


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
        optional_headers = ['anthropic-beta', 'user-agent', 'app-code']
        for header in optional_headers:
            if header in original_headers:
                headers[header] = original_headers[header]

        return headers

    def _remove_cache_control_recursive(self, data):
        """递归返回移除 cache_control 后的新数据结构"""
        if isinstance(data, dict):
            return {
                key: self._remove_cache_control_recursive(value)
                for key, value in data.items()
                if key != 'cache_control'
            }
        if isinstance(data, list):
            return [self._remove_cache_control_recursive(item) for item in data]
        return data

    def _standardize_cache_control(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化请求中的 cache_control 策略

        1. 递归清理所有位置的 cache_control（包括 system、messages 等）
        2. 只在最后一条消息的最后添加标准的 cache_control

        Args:
            request_data: 完整的请求数据

        Returns:
            处理后的请求数据
        """
        logger.info(f"Original request keys: {request_data.keys()}")

        # 第一步：递归删除所有位置的 cache_control
        standardized_request = self._remove_cache_control_recursive(request_data)

        # 第二步：在 messages 的最后一条消息添加 cache_control
        messages = standardized_request.get('messages', [])
        if messages:
            last_message = messages[-1]
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
                    last_message['content'] = [{
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
        logger.info(f"Standardized request keys: {standardized_request.keys()}")
        return standardized_request

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

        # 标准化缓存策略：清理所有 cache_control（包括 system、messages 等），
        # 只在最后一条消息添加标准配置
        standardized_request = self._standardize_cache_control(request_data)

        logger.info("Request processed with standardized cache_control strategy")

        # 直接转发请求给 Anthropic API
        response_data = await self._forward_to_anthropic(standardized_request, headers)
        return response_data

    async def handle_stream_request(self, request_data: Dict[str, Any],
                                  headers: Optional[Dict[str, str]] = None,
                                  request_id: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """处理流式 Anthropic API 请求"""
        headers = headers or {}

        # 验证请求格式
        if not self._validate_anthropic_request(request_data):
            raise ValueError("Invalid Anthropic request format")

        # 标准化缓存策略：清理所有 cache_control（包括 system、messages 等），
        # 只在最后一条消息添加标准配置
        standardized_request = self._standardize_cache_control(request_data)

        log_prefix = f"[request_id={request_id}] " if request_id else ""
        logger.info(f"{log_prefix}Stream request processed with standardized cache_control strategy")

        # 转发流式请求给 Anthropic API
        async for chunk in self._forward_stream_to_anthropic(standardized_request, headers, request_id=request_id):
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
                                      headers: Optional[Dict[str, str]] = None,
                                      request_id: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """转发流式请求到Anthropic API"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()
        url = self.messages_endpoint

        log_prefix = f"[request_id={request_id}] " if request_id else ""
        logger.info(f"{log_prefix}Forwarding stream request to: {url}")
        logger.info(f"{log_prefix}Request summary: {json.dumps(summarize_request_for_logging(request_data), ensure_ascii=False)}")


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

                logger.info(f"{log_prefix}Successfully connected to streaming endpoint")

                # 使用更健壮的流式读取方法
                try:
                    chunk_count = 0
                    total_bytes = 0
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        chunk_count += 1
                        total_bytes += len(chunk)
                        yield chunk
                except asyncio.CancelledError:
                    logger.warning(
                        f"{log_prefix}Streaming response cancelled, likely client disconnected "
                        f"after {chunk_count} chunks and {total_bytes} bytes"
                    )
                    raise
                except (BrokenPipeError, ConnectionResetError) as e:
                    logger.warning(
                        f"{log_prefix}Client connection dropped during streaming: {str(e)} "
                        f"after {chunk_count} chunks and {total_bytes} bytes"
                    )
                    return
                except (aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
                    logger.error(f"{log_prefix}Upstream streaming interrupted: {str(e)}")
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

                logger.info(
                    f"{log_prefix}Streaming request completed successfully "
                    f"with {chunk_count} chunks and {total_bytes} bytes"
                )

        except asyncio.TimeoutError:
            logger.error(f"{log_prefix}Timeout error from Anthropic API: stream request timed out")
            error_data = {
                "error": {
                    "type": "timeout_error",
                    "message": "Request to Anthropic API timed out"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except aiohttp.ClientResponseError as e:
            logger.error(f"{log_prefix}HTTP error from Anthropic API: {e.status} - {str(e)}")
            error_data = {
                "error": {
                    "type": "api_error",
                    "message": f"Anthropic API returned status {e.status}: {str(e)}"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except aiohttp.ClientError as e:
            logger.error(f"{log_prefix}Upstream request error: {str(e)}")
            error_data = {
                "error": {
                    "type": "network_error",
                    "message": f"Network error: {str(e)}"
                }
            }
            error_chunk = f'event: error\ndata: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except asyncio.CancelledError:
            logger.warning(f"{log_prefix}Streaming request cancelled before completion")
            raise

        except (BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"{log_prefix}Client connection dropped before stream completed: {str(e)}")
            return

        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error during streaming: {str(e)}")
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
        request_id = request.headers.get('x-request-id') or str(uuid.uuid4())[:8]

        # 检查是否为流式请求
        is_stream = request_data.get('stream', False)
        request_data.pop("top_p", None)
        request_data.pop("temperature", None)

        # 预处理模型名称：处理-thinking结尾的模型
        request_data = normalize_anthropic_model_and_thinking(request_data)

        # 获取请求头
        headers = dict(request.headers)

        # 处理请求
        logger.info(f"[request_id={request_id}] Received {'stream' if is_stream else 'non-stream'} request")
        logger.info(
            f"[request_id={request_id}] Request summary: "
            f"{json.dumps(summarize_request_for_logging(request_data), ensure_ascii=False)}"
        )

        if is_stream:
            # 流式回复
            logger.info(f"[request_id={request_id}] Processing as streaming request")
            return StreamingResponse(
                request_handler.handle_stream_request(request_data, headers, request_id=request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "X-Request-Id": request_id,
                }
            )
        else:
            # 非流式回复
            logger.info(f"[request_id={request_id}] Processing as non-streaming request")
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

                logger.warning(f"[request_id={request_id}] Request failed: {response_data}")
                return JSONResponse(status_code=status_code, content=response_data)

            logger.info(f"[request_id={request_id}] Request completed successfully")
            return JSONResponse(content=response_data, headers={"X-Request-Id": request_id})

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
