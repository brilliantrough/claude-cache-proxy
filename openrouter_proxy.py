import os
import json
import logging
import asyncio
import uuid
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
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


DEFAULT_OPENROUTER_MODEL_MAP = {
    "claude-opus-4-6": "anthropic/claude-opus-4.6",
    "claude-opus-latest": "anthropic/claude-opus-4.6",
    "claude-sonnet-latest": "anthropic/claude-sonnet-4.6",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4.5",
    "claude-opus-4-5-20251101": "anthropic/claude-opus-4.5",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
    "claude-opus-4-1-20250805": "anthropic/claude-opus-4.1",
}

DEFAULT_OPENROUTER_MODEL_MAP_FILE = "openrouter_model_map.json"


def _summarize_content(content: Any) -> Dict[str, Any]:
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

        for block in content:
            if not isinstance(block, dict):
                block_types.append(type(block).__name__)
                continue

            block_type = block.get('type', 'unknown')
            block_types.append(block_type)
            if block_type in ('image', 'image_url'):
                image_count += 1
            elif block_type == 'text':
                text_blocks += 1

        return {
            "content_type": "blocks",
            "block_count": len(content),
            "block_types": block_types,
            "image_count": image_count,
            "text_blocks": text_blocks,
        }

    return {
        "content_type": type(content).__name__
    }


def summarize_request_for_logging(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成安全的请求摘要，避免日志输出完整消息内容"""
    summary = {
        "original_model": request_data.get("original_model", request_data.get("model")),
        "model": request_data.get("model"),
        "stream": request_data.get("stream", False),
        "max_tokens": request_data.get("max_tokens"),
        "temperature": request_data.get("temperature"),
        "top_p": request_data.get("top_p"),
        "tools_count": len(request_data.get("tools", [])) if isinstance(request_data.get("tools"), list) else None,
    }

    messages = request_data.get("messages", [])
    if isinstance(messages, list):
        summary["messages_count"] = len(messages)
        summary["message_roles"] = [message.get("role", "unknown") for message in messages if isinstance(message, dict)]

        if messages and isinstance(messages[-1], dict):
            summary["last_message"] = {
                "role": messages[-1].get("role", "unknown"),
                **_summarize_content(messages[-1].get("content"))
            }

    thinking = request_data.get("thinking")
    if isinstance(thinking, dict):
        summary["thinking"] = {
            "type": thinking.get("type"),
            "budget_tokens": thinking.get("budget_tokens")
        }

    reasoning = request_data.get("reasoning")
    if isinstance(reasoning, dict):
        summary["reasoning"] = {
            "enabled": reasoning.get("enabled"),
            "max_tokens": reasoning.get("max_tokens")
        }

    return summary


def _normalize_model_map(raw_model_map: Any, source: str) -> Dict[str, str]:
    """标准化模型映射配置，忽略空 key/value"""
    if not isinstance(raw_model_map, dict):
        logger.warning(f"{source} must be a JSON object, ignoring mapping")
        return {}

    return {
        str(alias).strip(): str(target).strip()
        for alias, target in raw_model_map.items()
        if str(alias).strip() and str(target).strip()
    }


def load_model_map_from_env() -> Dict[str, str]:
    """加载模型映射：默认值 < JSON 文件 < 环境变量"""
    model_map = DEFAULT_OPENROUTER_MODEL_MAP.copy()

    model_map_file = os.getenv("OPENROUTER_MODEL_MAP_FILE", DEFAULT_OPENROUTER_MODEL_MAP_FILE)
    if model_map_file and os.path.exists(model_map_file):
        try:
            with open(model_map_file, "r", encoding="utf-8") as file:
                file_model_map = json.load(file)
            model_map.update(_normalize_model_map(file_model_map, model_map_file))
            logger.info(f"Loaded OpenRouter model map from {model_map_file}")
        except json.JSONDecodeError as error:
            logger.warning(f"Failed to parse {model_map_file}: {error}")
        except OSError as error:
            logger.warning(f"Failed to read {model_map_file}: {error}")

    raw_model_map = os.getenv("OPENROUTER_MODEL_MAP_JSON")

    if not raw_model_map:
        return model_map

    try:
        custom_model_map = json.loads(raw_model_map)
        model_map.update(_normalize_model_map(custom_model_map, "OPENROUTER_MODEL_MAP_JSON"))
    except json.JSONDecodeError as error:
        logger.warning(f"Failed to parse OPENROUTER_MODEL_MAP_JSON: {error}")

    return model_map

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
        self.model_map = load_model_map_from_env()
        self.default_reasoning_budget = int(os.getenv('OPENROUTER_REASONING_BUDGET', '30000'))
        self.default_thinking_max_tokens = int(os.getenv('OPENROUTER_THINKING_MAX_TOKENS', '80000'))

        # 确保 API URL 以 /v1 结尾，如果不是则添加
        if not self.api_url.endswith('/v1'):
            self.api_url = f"{self.api_url}/v1"

        # 构建聊天完成端点
        self.chat_endpoint = f"{self.api_url}/chat/completions"

        logger.info(f"Initialized OpenAI Request Handler with cache_control support")
        logger.info(f"Target API URL: {self.api_url}")

    def _normalize_model_and_reasoning(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化模型名，并根据 -thinking 后缀启用 reasoning"""
        normalized_request = request_data.copy()

        original_model = normalized_request.get('model', '')
        if not isinstance(original_model, str) or not original_model.strip():
            return normalized_request

        model_name = original_model.strip()
        thinking_enabled_by_suffix = False

        if model_name.endswith('-thinking'):
            model_name = model_name[:-len('-thinking')]
            thinking_enabled_by_suffix = True

        resolved_model = self.model_map.get(model_name, model_name)
        normalized_request['original_model'] = original_model
        normalized_request['model'] = resolved_model

        if thinking_enabled_by_suffix:
            existing_thinking = normalized_request.get('thinking', {})
            budget_tokens = self.default_reasoning_budget
            if isinstance(existing_thinking, dict) and isinstance(existing_thinking.get('budget_tokens'), int):
                budget_tokens = existing_thinking['budget_tokens']

            normalized_request['thinking'] = {
                'type': 'enabled',
                'budget_tokens': budget_tokens
            }

            max_tokens = normalized_request.get('max_tokens')
            if not isinstance(max_tokens, int) or max_tokens < self.default_thinking_max_tokens:
                normalized_request['max_tokens'] = self.default_thinking_max_tokens

        return normalized_request

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
                )
            )
            self.proxy_url = proxy_url  # 保存代理URL供请求使用
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

        normalized_request = self._normalize_model_and_reasoning(request_data)
        messages = normalized_request.get('messages', [])

        # 添加 cache_control 到最后一条消息
        cached_messages = self._add_cache_control_to_messages(messages)

        # 更新请求中的消息
        cached_request = normalized_request.copy()
        cached_request['messages'] = cached_messages

        # 转换 thinking 字段为 reasoning 字段（OpenRouter 格式）
        claude_reasoning = cached_request.get("thinking", {"type": "disabled", "budget_tokens": self.default_reasoning_budget})
        cached_request['reasoning'] = {
            "enabled": claude_reasoning.get('type') == "enabled",
            "max_tokens": claude_reasoning.get('budget_tokens', self.default_reasoning_budget)
        }

        logger.info("Request processed with OpenRouter cache_control strategy")

        # 转发请求给目标API
        response_data = await self._forward_to_api(cached_request, headers)
        return response_data

    async def handle_stream_request(self, request_data: Dict[str, Any],
                                  headers: Optional[Dict[str, str]] = None,
                                  request_id: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """处理流式 OpenAI API 请求"""
        headers = headers or {}

        # 验证请求格式
        if not self._validate_openai_request(request_data):
            raise ValueError("Invalid OpenAI request format")

        normalized_request = self._normalize_model_and_reasoning(request_data)
        messages = normalized_request.get('messages', [])

        # 添加 cache_control 到最后一条消息
        cached_messages = self._add_cache_control_to_messages(messages)

        # 更新请求中的消息
        cached_request = normalized_request.copy()
        cached_request['messages'] = cached_messages

        # 转换 thinking 字段为 reasoning 字段（OpenRouter 格式）
        claude_reasoning = cached_request.get("thinking", {"type": "disabled", "budget_tokens": self.default_reasoning_budget})
        cached_request['reasoning'] = {
            "enabled": claude_reasoning.get('type') == "enabled",
            "max_tokens": claude_reasoning.get('budget_tokens', self.default_reasoning_budget)
        }

        log_prefix = f"[request_id={request_id}] " if request_id else ""
        logger.info(f"{log_prefix}Stream request processed with OpenRouter cache_control strategy")

        # 转发流式请求给目标API
        async for chunk in self._forward_stream_to_api(cached_request, headers, request_id=request_id):
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
            # 在请求级别设置代理
            kwargs = {
                "json": request_data,
                "headers": prepared_headers
            }
            if hasattr(self, 'proxy_url') and self.proxy_url:
                kwargs["proxy"] = self.proxy_url

            async with session.post(url, **kwargs) as response:
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
                                    headers: Optional[Dict[str, str]] = None,
                                    request_id: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """转发流式请求到目标API"""
        prepared_headers = self._prepare_headers(headers or {})
        session = await self._get_session()
        url = self.chat_endpoint

        log_prefix = f"[request_id={request_id}] " if request_id else ""
        logger.info(f"{log_prefix}Forwarding stream request to: {url}")
        logger.info(f"{log_prefix}Request summary: {json.dumps(summarize_request_for_logging(request_data), ensure_ascii=False)}")

        try:
            # 在请求级别设置代理
            kwargs = {
                "json": request_data,
                "headers": prepared_headers,
                # 增加流式响应的缓冲区大小
                "read_bufsize": 8192,
                # 禁用自动解压缩，处理原始流
                "auto_decompress": False,
            }
            if hasattr(self, 'proxy_url') and self.proxy_url:
                kwargs["proxy"] = self.proxy_url

            async with session.post(url, **kwargs) as response:
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
                    error_chunk = f'data: {json.dumps(error_data)}\n\n'.encode('utf-8')
                    yield error_chunk
                    return

                logger.info(
                    f"{log_prefix}Streaming request completed successfully "
                    f"with {chunk_count} chunks and {total_bytes} bytes"
                )

        except aiohttp.ClientResponseError as e:
            logger.error(f"{log_prefix}HTTP error during streaming: {e.status} - {e.message}")
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
            logger.error(f"{log_prefix}Timeout during streaming: {str(e)}")
            # 发送超时错误信息
            error_data = {
                "error": {
                    "type": "timeout_error",
                    "message": f"Request timeout: {str(e)}"
                }
            }
            error_chunk = f'data: {json.dumps(error_data)}\n\n'.encode('utf-8')
            yield error_chunk

        except asyncio.CancelledError:
            logger.warning(f"{log_prefix}Streaming request cancelled before completion")
            raise

        except (BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"{log_prefix}Client connection dropped before stream completed: {str(e)}")
            return

        except Exception as e:
            logger.error(f"{log_prefix}Stream request error: {str(e)}")
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
        request_id = request.headers.get('x-request-id') or str(uuid.uuid4())[:8]

        # 检查是否为流式请求
        is_stream = request_data.get('stream', False)

        # 获取请求头
        headers = dict(request.headers)

        # 处理请求
        logger.info(f"[request_id={request_id}] Received {'stream' if is_stream else 'non-stream'} chat completion request")
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
                logger.warning(f"[request_id={request_id}] Request failed: {response_data}")
                return JSONResponse(status_code=500, content=response_data)

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
