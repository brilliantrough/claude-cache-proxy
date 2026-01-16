import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from dotenv import load_dotenv
import aiohttp

# 加载环境变量，命令行环境变量优先级更高
load_dotenv()  # 不覆盖现有环境变量

# 配置日志
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="OpenAI Format Proxy",
    description="A proxy server for OpenAI API with cache_control support",
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
request_handler: Optional['OpenAIRequestHandler'] = None


class OpenAIRequestHandler:
    """OpenAI API 请求处理器，负责转发请求并添加 cache_control 支持"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

        # 确保 API URL 以 /v1 结尾，如果不是则添加
        if not self.api_url.endswith('/v1'):
            self.api_url = f"{self.api_url}/v1"

        # 构建聊天完成端点
        self.chat_endpoint = f"{self.api_url}/chat/completions"

        logger.info(f"Initialized OpenAI Request Handler with cache_control support")
        logger.info(f"Target API URL: {self.api_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp 会话"""
        if self.session is None or self.session.closed:
            # 增加超时时间以处理长上下文
            timeout = aiohttp.ClientTimeout(
                total=300,  # 总超时增加到5分钟
                connect=60,  # 连接超时1分钟
                sock_read=240  # 读取超时4分钟
            )

            # 检查代理设置
            proxy_url = (
                os.environ.get('HTTPS_PROXY') or
                os.environ.get('HTTP_PROXY') or
                os.environ.get('ALL_PROXY')
            )

            if proxy_url:
                logger.info(f"Using proxy: {proxy_url}")

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
                    proxy=proxy_url,
                )
            )
        return self.session

    def _prepare_headers(self, original_headers: Dict[str, str]) -> Dict[str, str]:
        """准备请求头，使用配置中的默认API Key"""
        headers = {}

        # 使用标准的 Authorization 头（大写 A，与 OpenRouter 官方示例一致）
        headers['Authorization'] = f'Bearer {self.api_key}'
        headers['Content-Type'] = 'application/json'
        # 添加自定义header
        headers['HTTP-Referer'] = 'https://api.pezayo.com'
        headers['X-Title'] = 'One hub 站点'
        logger.info("Using configured default API key")

        return headers

    def _add_cache_control_to_messages(self, messages: list) -> list:
        """为消息添加 cache_control，遵循 OpenRouter 规范"""
        if not messages:
            return messages

        standardized_messages = []

        for message in messages:
            cleaned_message = message.copy()

            # 清理现有的 cache_control
            if 'cache_control' in cleaned_message:
                del cleaned_message['cache_control']

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

            standardized_messages.append(cleaned_message)

        # 在最后一条消息添加标准的 cache_control
        if standardized_messages:
            last_message = standardized_messages[-1]
            content = last_message.get('content', [])

            if isinstance(content, list):
                if content:
                    # 如果有 content 块，在最后一个块中添加 cache_control
                    if isinstance(content[-1], dict):
                        content[-1]['cache_control'] = {"type": "ephemeral"}
                    else:
                        # 如果最后一个块不是 dict，创建新的 dict 块
                        content.append({
                            "type": "text",
                            "text": str(content[-1]) if content[-1] else "",
                            "cache_control": {"type": "ephemeral"}
                        })
                else:
                    # 如果 content 为空，创建新的 content 块
                    content = [{
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral"}
                    }]
            elif isinstance(content, str):
                # 将字符串转换为 list 格式并添加 cache_control
                last_message['content'] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                # 如果 content 为 None 或其他格式，创建新的 content
                last_message['content'] = [
                    {
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]

        logger.info(f"Added cache_control to last message (no TTL for OpenRouter)")
        return standardized_messages

    def _validate_openai_request(self, request_data: Dict[str, Any]) -> bool:
        """验证OpenAI请求格式"""
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
        """处理OpenAI API请求"""
        headers = headers or {}

        # 验证请求格式
        if not self._validate_openai_request(request_data):
            raise ValueError("Invalid OpenAI request format")

        messages = request_data.get('messages', [])

        # 添加 cache_control 到最后一条消息
        cached_messages = self._add_cache_control_to_messages(messages)

        # 更新请求中的消息
        cached_request = request_data.copy()
        cached_request['messages'] = cached_messages

        logger.info("Request processed with OpenRouter cache_control strategy")

        # 转发请求给目标API
        response_data = await self._forward_to_api(cached_request, headers)
        return response_data

    async def handle_stream_request(self, request_data: Dict[str, Any],
                                  headers: Optional[Dict[str, str]] = None) -> AsyncGenerator[bytes, None]:
        """处理流式 OpenAI API 请求"""
        headers = headers or {}

        # 验证请求格式
        if not self._validate_openai_request(request_data):
            raise ValueError("Invalid OpenAI request format")

        messages = request_data.get('messages', [])

        # 添加 cache_control 到最后一条消息
        cached_messages = self._add_cache_control_to_messages(messages)

        # 更新请求中的消息
        cached_request = request_data.copy()
        cached_request['messages'] = cached_messages

        # 转换 thinking 字段为 reasoning 字段（OpenRouter 格式）
        claude_reasoning = cached_request.get("thinking", {"type": "disabled", "budget_tokens": 3276})
        cached_request['reasoning'] = {"enabled": claude_reasoning['type'] == "enabled", "max_tokens": claude_reasoning['budget_tokens']}

        logger.info("Stream request processed with OpenRouter cache_control strategy")

        # 转发流式请求给目标API
        async for chunk in self._forward_stream_to_api(cached_request, headers):
            yield chunk

    async def handle_models_request(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """处理models端点请求"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()

        # 构建完整的URL
        if self.api_url.endswith('/v1'):
            models_url = f"{self.api_url}/models"
        else:
            models_url = f"{self.api_url}/v1/models"

        logger.info(f"Forwarding models request to: {models_url}")

        try:
            async with session.get(models_url, headers=prepared_headers) as response:
                if response.headers.get('content-type', '').startswith('application/json'):
                    return await response.json()
                else:
                    return {"data": await response.text()}

        except Exception as e:
            logger.error(f"Models request error: {str(e)}")
            return {
                "error": {
                    "type": "api_error",
                    "message": f"Models request failed: {str(e)}"
                }
            }

    async def _forward_to_api(self, request_data: Dict[str, Any],
                             headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """转发请求到目标API"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()
        url = self.chat_endpoint

        logger.info(f"Forwarding request to: {url}")

        try:
            async with session.post(
                url,
                json=request_data,
                headers=prepared_headers
            ) as response:
                response.raise_for_status()

                response_data = await response.json()
                logger.info(f"Successfully received response from API")
                return response_data

        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return {
                "error": {
                    "type": "api_error",
                    "message": f"API request failed: {str(e)}"
                }
            }

    async def _forward_stream_to_api(self, request_data: Dict[str, Any],
                                    headers: Optional[Dict[str, str]] = None) -> AsyncGenerator[bytes, None]:
        """转发流式请求到目标API"""
        prepared_headers = self._prepare_headers(headers or {})
        logger.info(f"{prepared_headers}")
        session = await self._get_session()
        url = self.chat_endpoint

        logger.info(f"Forwarding stream request to: {url}")

        try:
            async with session.post(
                url,
                json=request_data,
                headers=prepared_headers,
                # 增加流式响应的缓冲区大小
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
                    error_chunk = f'data: {json.dumps(error_data)}\n\n'.encode('utf-8')
                    yield error_chunk
                    return

                logger.info(f"Streaming request completed successfully")

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error during streaming: {e.status} - {e.message}")
            # 发送HTTP错误信息
            error_data = {
                "error": {
                    "type": "http_error",
                    "message": f"HTTP {e.status}: {e.message}"
                }
            }
            error_chunk = f'data: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            logger.error(f"Timeout during streaming: {str(e)}")
            # 发送超时错误信息
            error_data = {
                "error": {
                    "type": "timeout_error",
                    "message": f"Request timeout: {str(e)}"
                }
            }
            error_chunk = f'data: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except Exception as e:
            logger.error(f"Stream request error: {str(e)}")
            # 发送通用错误信息
            error_data = {
                "error": {
                    "type": "api_error",
                    "message": f"Streaming request failed: {str(e)}"
                }
            }
            error_chunk = f'data: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

    async def close(self):
        """清理资源"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("OpenAI HTTP session closed")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global request_handler

    # 读取配置
    api_url = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1')
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    # 初始化请求处理器
    request_handler = OpenAIRequestHandler(api_url, api_key)

    logger.info("OpenAI Format Proxy server started")
    logger.info(f"Target API URL: {api_url}")

    # 显示API Key信息（只显示前几位和后几位）
    if len(api_key) > 20:
        masked_key = f"{api_key[:10]}...{api_key[-10:]}"
    else:
        masked_key = f"{api_key[:5]}..."
    logger.info(f"Using API Key: {masked_key}")

    # 显示配置来源
    env_file_loaded = os.path.exists('.env')
    if env_file_loaded:
        logger.info("Configuration loaded from .env file (command line vars take priority)")
    else:
        logger.warning("No .env file found, using environment variables")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    if request_handler:
        await request_handler.close()
    logger.info("OpenAI Format Proxy server shutdown")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "OpenAI Format Proxy",
        "status": "running",
        "endpoints": {
            "/": "Service information",
            "/v1/chat/completions": "OpenAI compatible chat completions endpoint (POST)",
            "/v1/models": "OpenAI compatible models endpoint (GET)",
            "/health": "Health check"
        },
        "cache_control": "Enabled for OpenRouter (no TTL parameter)"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "openai-format-proxy"}


@app.get("/v1/models")
async def models_endpoint(request: Request):
    """OpenAI 兼容的模型端点"""
    if not request_handler:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        headers = dict(request.headers)
        response_data = await request_handler.handle_models_request(headers)

        if 'error' in response_data:
            logger.warning(f"Models request failed: {response_data}")
            return JSONResponse(status_code=500, content=response_data)

        logger.info("Models request completed successfully")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Unexpected error in models endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: Request):
    """OpenAI 兼容的聊天完成端点，支持流式和非流式回复"""
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

        logger.info(f"is_stream {is_stream}")

        # 获取请求头
        headers = dict(request.headers)
        logger.info(f"Header: {headers}")

        # 处理请求
        logger.info(f"Received {'stream' if is_stream else 'non-stream'} chat completion request")
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
                logger.warning(f"Request failed: {response_data}")
                return JSONResponse(status_code=500, content=response_data)

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
    host = os.getenv('OPENAI_PROXY_HOST', '0.0.0.0')
    port = int(os.getenv('OPENAI_PROXY_PORT', 9998))

    logger.info(f"Starting OpenAI Format Proxy server on {host}:{port}")

    # 启动服务器
    uvicorn.run(
        "openrouter_proxy:app",
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )
