"""
@file utils.py
@description 工具函数和辅助方法，为 Deep Research Agent 提供各种支持功能

主要功能：
- Tavily 搜索工具：tavily_search, tavily_search_async, summarize_webpage
- MCP 工具管理：load_mcp_tools, wrap_mcp_authenticate_tool, get_mcp_access_token
- Token 管理：is_token_limit_exceeded, get_model_token_limit, remove_up_to_last_ai_message
- 工具组装：get_all_tools, get_search_tool
- 辅助函数：get_today_str, get_api_key_for_model, get_tavily_api_key

依赖关系：
- tavily.AsyncTavilyClient: Tavily 搜索客户端
- langchain_mcp_adapters.client: MCP 客户端适配器
- langchain_core.tools: 工具定义基类

模块结构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                              utils.py 模块结构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Tavily 搜索工具 (第 51-220 行)                    │   │
│  │  - tavily_search: 主搜索工具，执行搜索并摘要结果                      │   │
│  │  - tavily_search_async: 异步执行多个搜索查询                         │   │
│  │  - summarize_webpage: 使用 AI 摘要网页内容                           │   │
│  │  - get_tavily_api_key: 获取 Tavily API 密钥                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MCP 工具管理 (第 220-450 行)                       │   │
│  │  - load_mcp_tools: 从 MCP 服务器加载工具                             │   │
│  │  - wrap_mcp_authenticate_tool: 包装需要认证的 MCP 工具               │   │
│  │  - get_mcp_access_token: 获取 MCP 访问令牌                          │   │
│  │  - fetch_tokens: 从存储中获取令牌                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Token 管理 (第 450-550 行)                         │   │
│  │  - is_token_limit_exceeded: 检查是否超出 token 限制                  │   │
│  │  - get_model_token_limit: 获取模型的 token 限制                      │   │
│  │  - remove_up_to_last_ai_message: 移除消息以减少 token 使用           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    工具组装 (第 550-700 行)                           │   │
│  │  - get_all_tools: 获取所有可用工具（搜索 + MCP + think_tool）        │   │
│  │  - get_search_tool: 根据配置获取搜索工具                             │   │
│  │  - think_tool: 战略思考工具                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    辅助函数 (第 700-930 行)                           │   │
│  │  - get_today_str: 获取当前日期字符串                                  │   │
│  │  - get_api_key_for_model: 根据模型获取 API 密钥                      │   │
│  │  - get_notes_from_tool_calls: 从工具调用中提取笔记                   │   │
│  │  - openai_websearch_called: 检查是否调用了 OpenAI 原生搜索           │   │
│  │  - anthropic_websearch_called: 检查是否调用了 Anthropic 原生搜索     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

# ==================== Tavily 搜索工具 ====================
# 这部分提供了基于 Tavily API 的网络搜索功能
# Tavily 是一个专为 AI 应用优化的搜索 API，提供高质量的搜索结果

# Tavily 搜索工具的描述，会显示给 LLM
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """从 Tavily 搜索 API 获取并摘要搜索结果。

    这是主要的搜索工具，执行以下步骤：
    1. 异步执行多个搜索查询
    2. 去重搜索结果（按 URL）
    3. 使用 AI 模型摘要网页内容
    4. 格式化并返回最终结果

    Args:
        queries: 要执行的搜索查询列表
        max_results: 每个查询返回的最大结果数（默认 5）
        topic: 搜索结果的主题过滤器（general/news/finance）
        config: 运行时配置，包含 API 密钥和模型设置

    Returns:
        格式化的字符串，包含摘要后的搜索结果
    """
    # ========== 步骤 1: 异步执行搜索查询 ==========
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )

    # ========== 步骤 2: 按 URL 去重搜索结果 ==========
    # 避免处理相同的内容多次，提高效率
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # ========== 步骤 3: 配置摘要模型 ==========
    configurable = Configuration.from_runnable_config(config)

    # 字符限制：确保不超过模型的 token 限制（可配置）
    max_char_to_include = configurable.max_content_length

    # 初始化摘要模型，包含重试逻辑
    # 这确保了即使摘要失败也能继续处理
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # ========== 步骤 4: 创建摘要任务 ==========
    # 对于没有原始内容的结果，使用 noop 函数跳过
    async def noop():
        """空操作函数，用于处理没有原始内容的结果。"""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # ========== 步骤 5: 并行执行所有摘要任务 ==========
    # 使用 asyncio.gather 实现并发，提高性能
    summaries = await asyncio.gather(*summarization_tasks)

    # ========== 步骤 6: 合并结果和摘要 ==========
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # ========== 步骤 7: 格式化最终输出 ==========
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output

async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """异步执行多个 Tavily 搜索查询。

    这个函数并发执行多个搜索查询，提高效率。

    Args:
        search_queries: 要执行的搜索查询字符串列表
        max_results: 每个查询返回的最大结果数（默认 5）
        topic: 结果过滤的主题类别（general/news/finance）
        include_raw_content: 是否包含完整的网页内容（默认 True）
        config: 运行时配置，用于获取 API 密钥

    Returns:
        Tavily API 返回的搜索结果字典列表
    """
    # ========== 初始化 Tavily 客户端 ==========
    # 从配置中获取 API 密钥
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    # ========== 创建并发搜索任务 ==========
    # 为每个查询创建一个搜索任务，以便并发执行
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    # ========== 并行执行所有搜索查询 ==========
    # asyncio.gather 等待所有任务完成并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型摘要网页内容，包含超时保护。

    这个函数使用配置的 AI 模型对网页内容进行摘要，
    提取关键信息和重要摘录。

    Args:
        model: 配置用于摘要的聊天模型
        webpage_content: 要摘要的原始网页内容

    Returns:
        格式化的摘要，包含关键摘录；如果摘要失败则返回原始内容
    """
    try:
        # ========== 创建摘要提示 ==========
        # 使用预定义的提示模板，包含当前日期上下文
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        # ========== 执行摘要，包含超时保护 ==========
        # 60 秒超时防止摘要过程无限期挂起
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0
        )

        # ========== 格式化摘要输出 ==========
        # 使用结构化的 XML 标签组织摘要内容
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # ========== 超时处理 ==========
        # 如果摘要超过 60 秒，记录警告并返回原始内容
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content

    except Exception as e:
        # ========== 其他错误处理 ==========
        # 捕获摘要过程中的其他异常，记录并返回原始内容
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

# ==================== 战略思考工具 ====================
# think_tool 允许 Agent 在研究过程中进行战略性反思
# 这是一个"伪工具"，不执行实际操作，只是记录反思内容
# 目的是让 Agent 在搜索之间暂停思考，提高研究质量

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """用于研究进展和决策制定的战略反思工具。

    这个工具允许 Agent 在搜索之间暂停并进行战略性思考。
    它是一个"伪工具"，不执行实际操作，只是记录反思内容，
    目的是提高研究质量和决策的系统性。

    使用场景：
    - 接收搜索结果后：我找到了哪些关键信息？
    - 决定下一步之前：我有足够的信息来全面回答吗？
    - 评估研究差距时：我还缺少哪些具体信息？
    - 结束研究前：我现在能提供完整的答案吗？

    反思应该涵盖以下内容：
    1. 当前发现分析 - 我收集了哪些具体信息？
    2. 差距评估 - 还缺少哪些关键信息？
    3. 质量评估 - 我有足够的证据/示例来提供好的答案吗？
    4. 战略决策 - 应该继续搜索还是提供答案？

    Args:
        reflection: 关于研究进展、发现、差距和下一步的详细反思

    Returns:
        确认反思已被记录用于决策制定
    """
    return f"Reflection recorded: {reflection}"

# ==================== MCP 工具管理 ====================
# MCP (Model Context Protocol) 是一种协议，允许 LLM 与外部工具和服务交互
# 这部分提供了 MCP 工具的加载、认证和管理功能

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """使用 OAuth token 交换将 Supabase token 转换为 MCP 访问令牌。

    这个函数实现了 OAuth 2.0 token 交换流程，
    将用户的 Supabase 认证令牌交换为 MCP 服务器的访问令牌。

    Args:
        supabase_token: 有效的 Supabase 认证令牌
        base_mcp_url: MCP 服务器的基础 URL

    Returns:
        如果成功返回令牌数据字典，失败返回 None
    """
    try:
        # ========== 准备 OAuth token 交换请求 ==========
        # 按照 OAuth 2.0 token 交换规范构建请求数据
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        # ========== 执行 token 交换请求 ==========
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # ========== 成功获取令牌 ==========
                    token_data = await response.json()
                    return token_data
                else:
                    # ========== 记录错误详情用于调试 ==========
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")

    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None

async def get_tokens(config: RunnableConfig):
    """检索存储的认证令牌并验证过期时间。

    从存储中获取之前保存的令牌，并检查是否已过期。
    如果令牌已过期，则删除并返回 None。

    Args:
        config: 运行时配置，包含线程和用户标识符

    Returns:
        如果令牌有效且未过期返回令牌字典，否则返回 None
    """
    store = get_store()

    # ========== 从配置中提取必要的标识符 ==========
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    # ========== 从存储中检索令牌 ==========
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    # ========== 检查令牌过期时间 ==========
    # expires_in 表示令牌创建后的有效秒数
    expires_in = tokens.value.get("expires_in")
    created_at = tokens.created_at  # 令牌创建的时间
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # ========== 令牌已过期，清理并返回 None ==========
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """在配置存储中保存认证令牌。

    将令牌保存到存储中，以便后续使用。

    Args:
        config: 运行时配置，包含线程和用户标识符
        tokens: 要保存的令牌字典
    """
    store = get_store()

    # ========== 从配置中提取必要的标识符 ==========
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    # ========== 保存令牌到存储 ==========
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP 令牌，必要时获取新令牌。

    这个函数实现了令牌管理的完整流程：
    1. 首先尝试获取现有的有效令牌
    2. 如果没有有效令牌，使用 Supabase token 交换新令牌
    3. 保存新令牌以供后续使用

    Args:
        config: 运行时配置，包含认证详情

    Returns:
        有效的令牌字典，如果无法获取则返回 None
    """
    # ========== 步骤 1: 尝试获取现有的有效令牌 ==========
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    # ========== 步骤 2: 提取 Supabase token 用于新的 token 交换 ==========
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None

    # ========== 步骤 3: 提取 MCP 配置 ==========
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    # ========== 步骤 4: 使用 Supabase token 交换 MCP 令牌 ==========
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # ========== 步骤 5: 保存新令牌并返回 ==========
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """使用全面的认证和错误处理包装 MCP 工具。

    这个函数为 MCP 工具添加了错误处理层，特别是处理
    MCP 特定的错误（如需要交互的错误），并将其转换为
    用户友好的错误消息。

    Args:
        tool: 要包装的 MCP 结构化工具

    Returns:
        增强的工具，具有认证错误处理功能
    """
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        """增强的协程，包含 MCP 错误处理和用户友好的消息。"""

        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """递归搜索异常链中的 MCP 错误。"""
            if isinstance(exc, McpError):
                return exc

            # 处理 ExceptionGroup (Python 3.11+) 通过检查属性
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None

        try:
            # ========== 执行原始工具功能 ==========
            return await original_coroutine(**kwargs)

        except BaseException as original_error:
            # ========== 在异常链中搜索 MCP 特定错误 ==========
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 不是 MCP 错误，重新抛出原始异常
                raise original_error

            # ========== 处理 MCP 特定错误情况 ==========
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}

            # 检查需要交互的错误（错误代码 -32003）
            if error_code == -32003:
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"

                # 如果可用，提取用户友好的消息
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message

                # 如果提供了 URL，附加到错误消息中供用户参考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"

                raise ToolException(error_message) from original_error

            # 对于其他 MCP 错误，重新抛出原始异常
            raise original_error

    # ========== 用增强版本替换工具的协程 ==========
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置 MCP (Model Context Protocol) 工具，包含认证。

    这个函数从 MCP 服务器加载工具，处理认证，
    并将其配置为可供 Agent 使用。

    Args:
        config: 运行时配置，包含 MCP 服务器详情
        existing_tool_names: 已在使用的工具名称集合，用于避免冲突

    Returns:
        配置好的 MCP 工具列表，可供使用
    """
    configurable = Configuration.from_runnable_config(config)

    # ========== 步骤 1: 处理认证（如果需要） ==========
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None

    # ========== 步骤 2: 验证配置要求 ==========
    config_valid = (
        configurable.mcp_config and
        configurable.mcp_config.url and
        configurable.mcp_config.tools and
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )

    if not config_valid:
        return []

    # ========== 步骤 3: 设置 MCP 服务器连接 ==========
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"

    # 如果有令牌，配置认证头
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}

    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: 当 OAP 中合并多 MCP 服务器支持时，更新此代码

    # ========== 步骤 4: 从 MCP 服务器加载工具 ==========
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # 如果 MCP 服务器连接失败，返回空列表
        return []

    # ========== 步骤 5: 过滤并配置工具 ==========
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳过名称冲突的工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue

        # 只包含配置中指定的工具
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue

        # 用认证处理包装工具并添加到列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)

    return configured_tools


# ==================== 工具组装 ====================
# 这部分负责根据配置组装所有可用的工具
# 包括搜索工具、MCP 工具和战略思考工具

async def get_search_tool(search_api: SearchAPI):
    """根据指定的 API 提供商配置并返回搜索工具。

    支持多个搜索 API 提供商，包括 Anthropic、OpenAI、Tavily 等。

    Args:
        search_api: 要使用的搜索 API 提供商 (Anthropic/OpenAI/Tavily/None)

    Returns:
        为指定提供商配置的搜索工具对象列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # ========== Anthropic 原生网络搜索 ==========
        # 具有使用限制的 Anthropic 原生搜索功能
        return [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }]

    elif search_api == SearchAPI.OPENAI:
        # ========== OpenAI 网络搜索预览 ==========
        # OpenAI 的网络搜索预览功能
        return [{"type": "web_search_preview"}]

    elif search_api == SearchAPI.TAVILY:
        # ========== Tavily 搜索工具 ==========
        # 配置 Tavily 搜索工具及其元数据
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # ========== 无搜索功能 ==========
        # 未配置搜索功能
        return []

    # ========== 默认回退 ==========
    # 对于未知的搜索 API 类型
    return []
    
async def get_all_tools(config: RunnableConfig):
    """组装完整的工具包，包括研究、搜索和 MCP 工具。

    这个函数根据配置组装所有可用的工具，包括：
    1. 核心研究工具（ResearchComplete）
    2. 战略思考工具（think_tool）
    3. 搜索工具（根据配置的搜索 API）
    4. MCP 工具（如果配置）

    Args:
        config: 运行时配置，指定搜索 API 和 MCP 设置

    Returns:
        所有配置好的可用工具列表，可供研究操作使用
    """
    # ========== 步骤 1: 添加核心研究工具 ==========
    tools = [tool(ResearchComplete), think_tool]

    # ========== 步骤 2: 添加配置的搜索工具 ==========
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    # ========== 步骤 3: 跟踪现有工具名称以防止冲突 ==========
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in tools
    }

    # ========== 步骤 4: 添加 MCP 工具（如果配置） ==========
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)

    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具调用消息中提取研究笔记。

    这个函数从消息列表中过滤出工具调用结果，
    并提取其内容作为研究笔记。

    Args:
        messages: 消息列表，包含工具调用结果

    Returns:
        工具调用结果内容的列表
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

# ==================== 原生搜索检测 ====================
# 这部分用于检测是否使用了模型提供商的原生网络搜索功能
# OpenAI 和 Anthropic 都提供了原生的网络搜索能力

def anthropic_websearch_called(response):
    """检测 Anthropic 原生网络搜索是否在响应中被使用。

    通过检查响应元数据中的 server_tool_use 信息来判断
    是否调用了 Anthropic 的网络搜索功能。

    Args:
        response: Anthropic API 的响应对象

    Returns:
        如果调用了网络搜索返回 True，否则返回 False
    """
    try:
        # ========== 导航响应元数据结构 ==========
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # ========== 检查服务器端工具使用信息 ==========
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # ========== 查找网络搜索请求计数 ==========
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # ========== 如果有任何网络搜索请求，返回 True ==========
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # ========== 处理响应结构意外的情况 ==========
        return False

def openai_websearch_called(response):
    """检测 OpenAI 网络搜索功能是否在响应中被使用。

    通过检查响应的 additional_kwargs 中的 tool_outputs 来判断
    是否调用了 OpenAI 的网络搜索功能。

    Args:
        response: OpenAI API 的响应对象

    Returns:
        如果调用了网络搜索返回 True，否则返回 False
    """
    # ========== 检查响应元数据中的工具输出 ==========
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # ========== 在工具输出中查找网络搜索调用 ==========
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


# ==================== Token 限制管理 ====================
# 这部分用于检测和处理模型的 token/上下文长度限制
# 不同的模型提供商有不同的错误格式，需要分别处理

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断异常是否表示超出了 token/上下文限制。

    这个函数检查异常是否由于 token 或上下文长度限制而产生。
    不同的模型提供商（OpenAI、Anthropic、Google 等）有不同的错误格式，
    所以需要分别处理。

    Args:
        exception: 要分析的异常
        model_name: 可选的模型名称，用于优化提供商检测

    Returns:
        如果异常表示超出 token 限制返回 True，否则返回 False
    """
    error_str = str(exception).lower()

    # ========== 步骤 1: 从模型名称确定提供商（如果可用） ==========
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'

    # ========== 步骤 2: 检查提供商特定的 token 限制模式 ==========
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)

    # ========== 步骤 3: 如果提供商未知，检查所有提供商 ==========
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 OpenAI token 限制超出。

    OpenAI 使用特定的异常类型和错误代码来表示 token 限制。

    Args:
        exception: 要检查的异常
        error_str: 异常的小写字符串表示

    Returns:
        如果是 OpenAI token 限制错误返回 True
    """
    # ========== 分析异常元数据 ==========
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # ========== 检查这是否是 OpenAI 异常 ==========
    is_openai_exception = (
        'openai' in exception_type.lower() or
        'openai' in module_name.lower()
    )

    # ========== 检查典型的 OpenAI token 限制错误类型 ==========
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']

    if is_openai_exception and is_request_error:
        # 在错误消息中查找 token 相关的关键字
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # ========== 检查特定的 OpenAI 错误代码 ==========
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')

        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True

    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Anthropic token 限制超出。

    Anthropic 使用特定的错误消息来表示 token 限制。

    Args:
        exception: 要检查的异常
        error_str: 异常的小写字符串表示

    Returns:
        如果是 Anthropic token 限制错误返回 True
    """
    # ========== 分析异常元数据 ==========
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # ========== 检查这是否是 Anthropic 异常 ==========
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or
        'anthropic' in module_name.lower()
    )

    # ========== 检查 Anthropic 特定的错误模式 ==========
    is_bad_request = class_name == 'BadRequestError'

    if is_anthropic_exception and is_bad_request:
        # Anthropic 使用特定的错误消息表示 token 限制
        if 'prompt is too long' in error_str:
            return True

    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Google/Gemini token 限制超出。

    Google Gemini 使用资源耗尽错误来表示 token 限制。

    Args:
        exception: 要检查的异常
        error_str: 异常的小写字符串表示

    Returns:
        如果是 Gemini token 限制错误返回 True
    """
    # ========== 分析异常元数据 ==========
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # ========== 检查这是否是 Google/Gemini 异常 ==========
    is_google_exception = (
        'google' in exception_type.lower() or
        'google' in module_name.lower()
    )

    # ========== 检查 Google 特定的资源耗尽错误 ==========
    is_resource_exhausted = class_name in [
        'ResourceExhausted',
        'GoogleGenerativeAIFetchError'
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # ========== 检查特定的 Google API 资源耗尽模式 ==========
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True

    return False

# NOTE: 这个列表可能已过时或不适用于你的模型。请根据需要更新。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查找特定模型的 token 限制。

    这个函数在已知的模型 token 限制表中查找模型，
    返回该模型的最大 token 数。

    Args:
        model_string: 要查找的模型标识符字符串

    Returns:
        如果找到返回 token 限制整数，如果模型不在查找表中返回 None
    """
    # ========== 搜索已知的模型 token 限制 ==========
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit

    # ========== 模型未在查找表中找到 ==========
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通过移除最后一条 AI 消息之前的所有消息来截断消息历史。

    这个函数用于处理超出 token 限制的错误，通过移除最近的上下文来减少 token 使用。
    它从消息列表的末尾向前搜索，找到最后一条 AI 消息，
    然后返回该消息之前的所有消息（不包括该 AI 消息本身）。

    Args:
        messages: 要截断的消息对象列表

    Returns:
        截断后的消息列表，包含到最后一条 AI 消息之前的所有消息
    """
    # ========== 从后向前搜索最后一条 AI 消息 ==========
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # ========== 返回最后一条 AI 消息之前的所有内容 ==========
            # 不包括最后一条 AI 消息本身
            return messages[:i]

    # ========== 未找到 AI 消息，返回原始列表 ==========
    return messages

# ==================== 辅助函数 ====================
# 这部分提供各种通用的辅助功能

def get_today_str() -> str:
    """获取当前日期，格式化用于提示和输出中显示。

    返回人类可读的日期字符串，格式如 'Mon Jan 15, 2024'。

    Returns:
        格式化的日期字符串
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置中提取值，处理枚举和 None 值。

    这个函数处理不同类型的配置值，包括枚举、字符串、字典等。

    Args:
        value: 要提取的配置值

    Returns:
        提取后的值
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """获取特定模型的 API 密钥。

    根据配置，从环境变量或运行时配置中获取 API 密钥。
    支持 OpenAI、Anthropic 和 Google 等多个提供商。

    Args:
        model_name: 模型名称
        config: 运行时配置

    Returns:
        API 密钥字符串，如果未找到返回 None
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()

    if should_get_from_config.lower() == "true":
        # ========== 从运行时配置中获取 API 密钥 ==========
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        # ========== 从环境变量中获取 API 密钥 ==========
        if model_name.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """获取 Tavily API 密钥。

    根据配置，从环境变量或运行时配置中获取 Tavily API 密钥。

    Args:
        config: 运行时配置

    Returns:
        Tavily API 密钥字符串，如果未找到返回 None
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")

    if should_get_from_config.lower() == "true":
        # ========== 从运行时配置中获取 Tavily API 密钥 ==========
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        # ========== 从环境变量中获取 Tavily API 密钥 ==========
        return os.getenv("TAVILY_API_KEY")
