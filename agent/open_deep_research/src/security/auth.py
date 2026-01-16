"""
@file auth.py
@description 认证处理器，基于 Supabase 实现用户认证和资源访问控制

主要功能：
- get_current_user: 验证用户 JWT token
- on_thread_create: 创建线程时添加所有者元数据
- on_thread_read: 读取线程时过滤只显示用户自己的线程
- on_assistants_create/read: 助手资源的访问控制
- authorize_store: 存储资源的命名空间授权

依赖关系：
- langgraph_sdk.Auth: LangGraph 认证框架
- supabase: Supabase 客户端，用于 JWT 验证
"""

import os
import asyncio
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser
from supabase import create_client, Client
from typing import Optional, Any

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

# Auth 对象是一个容器，LangGraph 使用它来标记我们的认证函数
auth = Auth()


# authenticate 装饰器告诉 LangGraph 将此函数作为中间件调用
# 对每个请求进行处理。这将决定请求是否被允许
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """
    验证用户的 JWT token 是否有效。

    使用 Supabase 进行 JWT token 验证，确保用户身份合法。

    参数：
        authorization: 来自请求头的授权信息，格式为 "Bearer <token>"

    返回：
        包含用户身份信息的字典，格式为 {"identity": user_id}

    异常：
        HTTPException: 当授权头缺失、格式错误、token 无效或 Supabase 未初始化时抛出
    """

    # 确保存在授权头
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    # 解析授权头，提取 scheme 和 token
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    # 确保 Supabase 客户端已初始化
    if not supabase:
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Supabase client not initialized"
        )

    try:
        # 使用 asyncio.to_thread 在单独的线程中验证 JWT token，避免阻塞
        # 这将在后台线程中解码和验证 JWT token
        async def verify_token() -> dict[str, Any]:
            response = await asyncio.to_thread(supabase.auth.get_user, token)
            return response

        response = await verify_token()
        user = response.user

        # 检查用户是否存在
        if not user:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Invalid token or user not found"
            )

        # 如果 token 有效，返回用户信息
        return {
            "identity": user.id,
        }
    except Exception as e:
        # 处理来自 Supabase 的任何错误
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"Authentication error: {str(e)}"
        )


@auth.on.threads.create
@auth.on.threads.create_run
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """
    创建线程时添加所有者元数据。

    此处理器在创建新线程时运行，执行以下操作：
    1. 在创建的线程上设置元数据以跟踪所有权
    2. 确保只有创建者可以访问该线程

    参数：
        ctx: 认证上下文，包含当前用户信息
        value: 线程创建请求的值对象

    返回：
        无返回值，直接修改 value 对象中的元数据
    """

    # 如果是 Studio 用户，跳过处理（Studio 用户有特殊权限）
    if isinstance(ctx.user, StudioUser):
        return

    # 为正在创建的线程添加所有者元数据
    # 此元数据与线程一起存储并持久化
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.threads.read
@auth.on.threads.delete
@auth.on.threads.update
@auth.on.threads.search
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """
    仅允许用户读取自己的线程。

    此处理器在读取操作时运行。由于线程已经存在，我们不需要设置元数据，
    只需返回一个过滤器以确保用户只能看到自己的线程。

    参数：
        ctx: 认证上下文，包含当前用户信息
        value: 线程读取请求的值对象

    返回：
        包含所有者过滤条件的字典，格式为 {"owner": user_identity}
        如果是 Studio 用户则返回 None（不进行过滤）
    """
    # 如果是 Studio 用户，跳过处理（Studio 用户有特殊权限）
    if isinstance(ctx.user, StudioUser):
        return

    # 返回过滤条件，确保只能访问自己的线程
    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistants_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
):
    """
    创建助手时添加所有者元数据。

    此处理器在创建新助手时运行，为助手添加所有者元数据以跟踪所有权。

    参数：
        ctx: 认证上下文，包含当前用户信息
        value: 助手创建请求的值对象

    返回：
        无返回值，直接修改 value 对象中的元数据
    """
    # 如果是 Studio 用户，跳过处理（Studio 用户有特殊权限）
    if isinstance(ctx.user, StudioUser):
        return

    # 为正在创建的助手添加所有者元数据
    # 此元数据与助手一起存储并持久化
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.assistants.read
@auth.on.assistants.delete
@auth.on.assistants.update
@auth.on.assistants.search
async def on_assistants_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
):
    """
    仅允许用户读取自己的助手。

    此处理器在读取操作时运行。由于助手已经存在，我们不需要设置元数据，
    只需返回一个过滤器以确保用户只能看到自己的助手。

    参数：
        ctx: 认证上下文，包含当前用户信息
        value: 助手读取请求的值对象

    返回：
        包含所有者过滤条件的字典，格式为 {"owner": user_identity}
        如果是 Studio 用户则返回 None（不进行过滤）
    """

    # 如果是 Studio 用户，跳过处理（Studio 用户有特殊权限）
    if isinstance(ctx.user, StudioUser):
        return

    # 返回过滤条件，确保只能访问自己的助手
    return {"owner": ctx.user.identity}


@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    """
    存储资源的命名空间授权处理器。

    此处理器验证用户是否有权访问存储中的特定命名空间。
    命名空间是一个元组，可以看作是存储项的目录结构。

    参数：
        ctx: 认证上下文，包含当前用户信息
        value: 存储操作的值对象，包含 namespace 字段

    返回：
        无返回值，如果授权失败则抛出异常

    异常：
        AssertionError: 当用户尝试访问不属于自己的命名空间时抛出
    """
    # 如果是 Studio 用户，跳过处理（Studio 用户有特殊权限）
    if isinstance(ctx.user, StudioUser):
        return

    # "namespace" 字段是一个元组，可以看作是存储项的目录结构
    namespace: tuple = value["namespace"]
    # 验证命名空间的第一个元素是否为当前用户的身份，确保用户只能访问自己的命名空间
    assert namespace[0] == ctx.user.identity, "Not authorized"