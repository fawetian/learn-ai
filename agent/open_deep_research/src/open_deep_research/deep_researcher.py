"""
@file deep_researcher.py
@description Deep Research Agent 的主 LangGraph 工作流实现

主要功能：
- 定义完整的深度研究工作流（clarify → research_brief → supervisor → report）
- 实现三层架构：主图层、监督层（Supervisor）、研究层（Researcher）
- 支持并发研究任务执行
- 支持研究结果压缩和最终报告生成

核心节点：
- clarify_with_user: 判断是否需要向用户澄清问题
- write_research_brief: 将用户消息转换为结构化研究简报
- supervisor: 研究主管，规划研究策略，分配任务
- supervisor_tools: 执行监督工具调用，包括并发启动子研究员
- researcher: 子研究员，执行具体搜索任务
- researcher_tools: 处理研究员的工具调用
- compress_research: 压缩研究发现，保留关键信息
- final_report_generation: 生成最终综合报告

依赖关系：
- langgraph.graph: 状态图构建
- langchain.chat_models: 模型初始化
- open_deep_research.configuration: 配置管理
- open_deep_research.state: 状态定义
- open_deep_research.prompts: 提示词模板
- open_deep_research.utils: 工具函数

工作流架构图：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Deep Researcher Graph (主图)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START                                                                      │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────┐                                                    │
│  │ clarify_with_user   │ ──────────────────────────────────────────► END    │
│  │ (澄清用户需求)       │     (如果需要澄清，返回问题给用户)                  │
│  └─────────┬───────────┘                                                    │
│            │ (不需要澄清)                                                    │
│            ▼                                                                │
│  ┌─────────────────────┐                                                    │
│  │ write_research_brief│                                                    │
│  │ (生成研究简报)       │                                                    │
│  └─────────┬───────────┘                                                    │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Supervisor Subgraph (监督者子图)                   │   │
│  │  ┌──────────────┐      ┌───────────────────┐                        │   │
│  │  │  supervisor  │ ◄──► │ supervisor_tools  │                        │   │
│  │  │ (规划研究)    │      │ (执行工具调用)     │                        │   │
│  │  └──────────────┘      └─────────┬─────────┘                        │   │
│  │                                  │                                   │   │
│  │                    ┌─────────────┴─────────────┐                    │   │
│  │                    │  Researcher Subgraph ×N   │ (并发执行)          │   │
│  │                    │  ┌──────────┐ ┌────────┐  │                    │   │
│  │                    │  │researcher│►│tools   │  │                    │   │
│  │                    │  └──────────┘ └───┬────┘  │                    │   │
│  │                    │                   ▼       │                    │   │
│  │                    │  ┌─────────────────────┐  │                    │   │
│  │                    │  │ compress_research   │  │                    │   │
│  │                    │  └─────────────────────┘  │                    │   │
│  │                    └───────────────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────┐                                               │
│  │ final_report_generation │                                               │
│  │ (生成最终报告)           │                                               │
│  └─────────┬───────────────┘                                               │
│            │                                                                │
│            ▼                                                                │
│          END                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# ==================== 模型初始化 ====================
# 创建一个可配置的模型实例，支持动态切换模型、token 限制和 API key
# configurable_fields 允许在运行时通过 with_config() 修改这些参数
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


# ==================== 主图节点函数 ====================

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息并在研究范围不清楚时提出澄清问题。

    该函数判断用户的请求是否需要在进行研究前进行澄清。如果禁用澄清或不需要澄清，
    则直接进行研究。

    工作流程：
    1. 检查配置中是否启用澄清功能
    2. 如果禁用，直接跳转到研究简报生成
    3. 如果启用，使用模型分析用户消息是否需要澄清
    4. 根据分析结果返回澄清问题或验证消息

    Args:
        state: 当前代理状态，包含用户消息
        config: 运行时配置，包含模型设置和偏好

    Returns:
        Command 对象，指示返回澄清问题（END）或继续研究简报生成
    """
    # ========== 第一步：检查澄清功能是否启用 ==========
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进行研究
        return Command(goto="write_research_brief")

    # ========== 第二步：准备模型进行结构化澄清分析 ==========
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 配置模型以支持结构化输出和重试逻辑
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # ========== 第三步：分析是否需要澄清 ==========
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # ========== 第四步：根据分析结果进行路由 ==========
    if response.need_clarification:
        # 返回澄清问题给用户，结束当前流程
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 继续进行研究，返回验证消息
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化研究简报并初始化监督者。

    该函数分析用户消息并生成一份专注的研究简报，用于指导研究监督者。
    同时为监督者设置初始上下文，包括适当的提示词和指令。

    工作流程：
    1. 配置研究模型以支持结构化输出
    2. 使用模型将用户消息转换为结构化研究问题
    3. 初始化监督者的系统提示词和初始消息
    4. 返回命令以进入研究监督者阶段

    Args:
        state: 当前代理状态，包含用户消息
        config: 运行时配置，包含模型设置

    Returns:
        Command 对象，指示进入研究监督者阶段并传递初始化的监督者状态
    """
    # ========== 第一步：配置研究模型以支持结构化输出 ==========
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 配置模型以生成结构化研究问题
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # ========== 第二步：从用户消息生成结构化研究简报 ==========
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # ========== 第三步：初始化监督者的系统提示词和初始消息 ==========
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """研究主管，规划研究策略并将任务委派给研究员。

    监督者分析研究简报并决定如何将研究分解为可管理的任务。
    它可以使用 think_tool 进行战略规划，使用 ConductResearch 委派任务给子研究员，
    或使用 ResearchComplete 标记研究完成。

    工作流程：
    1. 配置监督者模型并绑定可用工具
    2. 根据当前上下文生成监督者响应
    3. 更新状态并进入工具执行阶段

    Args:
        state: 当前监督者状态，包含消息和研究上下文
        config: 运行时配置，包含模型设置

    Returns:
        Command 对象，指示进入监督者工具执行阶段
    """
    # ========== 第一步：配置监督者模型并绑定可用工具 ==========
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 可用工具：研究委派、完成信号和战略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # 配置模型以支持工具调用、重试逻辑和模型设置
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # ========== 第二步：根据当前上下文生成监督者响应 ==========
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # ========== 第三步：更新状态并进入工具执行阶段 ==========
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """
    执行监督者的工具调用，包括研究委派和战略思考

    这是监督者子图中最复杂的节点，负责处理三种类型的工具调用：
    1. think_tool - 战略性反思，帮助监督者规划研究策略
    2. ConductResearch - 委派研究任务给子研究员（并发执行）
    3. ResearchComplete - 标记研究完成，触发报告生成

    工作流程：
    1. 检查退出条件（迭代次数、无工具调用、研究完成）
    2. 处理 think_tool 调用（记录反思内容）
    3. 处理 ConductResearch 调用（并发启动研究员子图）
    4. 收集研究结果并返回给监督者

    Args:
        state: 当前监督者状态，包含消息历史和迭代计数
        config: 运行时配置，包含研究限制和模型设置

    Returns:
        Command 对象，指示下一步是继续监督循环还是结束研究阶段
    """
    # ========== 第一步：提取状态并检查退出条件 ==========
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 定义研究阶段的退出条件
    # 条件1：超过允许的最大迭代次数
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    # 条件2：模型没有调用任何工具（可能是模型认为信息已足够）
    no_tool_calls = not most_recent_message.tool_calls
    # 条件3：模型显式调用了 ResearchComplete 工具
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果满足任一退出条件，结束研究阶段
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                # 从所有工具调用中提取研究笔记
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # ========== 第二步：处理所有工具调用 ==========
    all_tool_messages = []  # 收集所有工具调用的响应消息
    update_payload = {"supervisor_messages": []}

    # ---------- 处理 think_tool 调用（战略反思）----------
    # think_tool 允许监督者在委派任务前进行战略性思考
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        # 提取反思内容并创建工具响应消息
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # ---------- 处理 ConductResearch 调用（研究委派）----------
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # 限制并发研究单元数量，防止资源耗尽和 API 速率限制
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            # ========== 并发执行研究任务 ==========
            # 为每个研究任务创建一个研究员子图实例
            # 使用 asyncio.gather 并发执行所有研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            # 等待所有研究任务完成
            tool_results = await asyncio.gather(*research_tasks)

            # 将研究结果转换为工具响应消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # 处理超出并发限制的研究请求（返回错误消息）
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # 聚合所有研究结果的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # ========== 错误处理 ==========
            # 如果遇到 token 限制或其他错误，结束研究阶段
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )

    # ========== 第三步：返回工具执行结果 ==========
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",  # 返回监督者节点继续规划
        update=update_payload
    ) 

# ==================== 监督者子图构建 ====================
# 监督者子图负责管理研究任务的分配和协调
# 它在 supervisor 和 supervisor_tools 之间循环，直到研究完成

# 创建监督者状态图，使用 SupervisorState 作为状态类型
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加监督者节点
supervisor_builder.add_node("supervisor", supervisor)           # 主监督者逻辑：规划研究策略
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行：处理研究委派

# 定义监督者工作流边
# START → supervisor：从入口点开始执行监督者逻辑
supervisor_builder.add_edge(START, "supervisor")

# 编译监督者子图，供主图调用
# 注意：supervisor_tools 会根据条件返回到 supervisor 或 END
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """执行特定主题的专注研究的个体研究员。

    研究员由监督者分配特定的研究主题，并使用可用工具（搜索、think_tool、MCP 工具）
    收集全面的信息。它可以在搜索之间使用 think_tool 进行战略规划。

    工作流程：
    1. 加载配置并验证工具可用性
    2. 配置研究员模型并绑定工具
    3. 根据系统上下文生成研究员响应
    4. 更新状态并进入工具执行阶段

    Args:
        state: 当前研究员状态，包含消息和主题上下文
        config: 运行时配置，包含模型设置和工具可用性

    Returns:
        Command 对象，指示进入研究员工具执行阶段
    """
    # ========== 第一步：加载配置并验证工具可用性 ==========
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # 获取所有可用的研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "未找到进行研究的工具：请配置搜索 API 或向配置中添加 MCP 工具。"
        )

    # ========== 第二步：配置研究员模型并绑定工具 ==========
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 准备系统提示词，如果可用则包含 MCP 上下文
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    # 配置模型以支持工具调用、重试逻辑和设置
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # ========== 第三步：根据系统上下文生成研究员响应 ==========
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # ========== 第四步：更新状态并进入工具执行阶段 ==========
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """安全执行工具，包含错误处理。

    该函数包装工具调用，捕获任何异常并返回错误消息，
    防止单个工具失败导致整个研究流程中断。

    Args:
        tool: 要执行的工具对象
        args: 工具的参数字典
        config: 运行时配置

    Returns:
        工具执行结果或错误消息字符串
    """
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"工具执行错误：{str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具，包括搜索工具和战略思考。

    该函数处理各种类型的研究员工具调用：
    1. think_tool - 战略反思，继续研究对话
    2. 搜索工具（tavily_search、web_search）- 信息收集
    3. MCP 工具 - 外部工具集成
    4. ResearchComplete - 标记个体研究任务完成

    工作流程：
    1. 提取当前状态并检查早期退出条件
    2. 处理其他工具调用（搜索、MCP 工具等）
    3. 检查后期退出条件（迭代次数、研究完成）
    4. 返回继续研究循环或进入压缩阶段的命令

    Args:
        state: 当前研究员状态，包含消息和迭代计数
        config: 运行时配置，包含研究限制和工具设置

    Returns:
        Command 对象，指示继续研究循环或进入压缩阶段
    """
    # ========== 第一步：提取当前状态并检查早期退出条件 ==========
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 检查是否进行了工具调用（包括原生网络搜索）
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or
        anthropic_websearch_called(most_recent_message)
    )

    # 如果没有工具调用和原生搜索，直接进入压缩阶段
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # ========== 第二步：处理其他工具调用（搜索、MCP 工具等）==========
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 并行执行所有工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 从执行结果创建工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # ========== 第三步：检查后期退出条件（处理工具后）==========
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩阶段
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # ========== 第四步：继续研究循环 ==========
    # 将工具结果返回给研究员，继续研究
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究发现压缩并综合为简洁的结构化摘要。

    该函数获取研究员工作中的所有研究发现、工具输出和 AI 消息，
    并将其提炼为清晰、全面的摘要，同时保留所有重要信息和发现。

    工作流程：
    1. 配置压缩模型
    2. 准备用于压缩的消息
    3. 使用重试逻辑尝试压缩，处理 token 限制问题
    4. 返回压缩后的研究摘要和原始笔记

    Args:
        state: 当前研究员状态，包含累积的研究消息
        config: 运行时配置，包含压缩模型设置

    Returns:
        包含压缩研究摘要和原始笔记的字典
    """
    # ========== 第一步：配置压缩模型 ==========
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })

    # ========== 第二步：准备用于压缩的消息 ==========
    researcher_messages = state.get("researcher_messages", [])

    # 添加指令以从研究模式切换到压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # ========== 第三步：使用重试逻辑尝试压缩 ==========
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # 创建专注于压缩任务的系统提示词
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)

            # 从所有工具和 AI 消息中提取原始笔记
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # 返回成功的压缩结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }

        except Exception as e:
            synthesis_attempts += 1

            # 通过删除较早的消息来处理超出 token 限制的情况
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # 对于其他错误，继续重试
            continue

    # ========== 第四步：如果所有尝试都失败，返回错误结果 ==========
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "错误：综合研究报告失败，已达到最大重试次数",
        "raw_notes": [raw_notes_content]
    }

# ==================== 研究员子图构建 ====================
# 研究员子图负责执行具体的搜索和信息收集任务
# 每个研究员实例处理一个特定的研究主题

# 创建研究员状态图
# - ResearcherState: 输入状态类型
# - ResearcherOutputState: 输出状态类型（只返回压缩后的研究结果）
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,  # 定义输出状态，过滤掉内部状态
    config_schema=Configuration
)

# 添加研究员节点
researcher_builder.add_node("researcher", researcher)                 # 主研究员逻辑：执行搜索
researcher_builder.add_node("researcher_tools", researcher_tools)     # 工具执行：处理搜索结果
researcher_builder.add_node("compress_research", compress_research)   # 研究压缩：整理研究发现

# 定义研究员工作流边
researcher_builder.add_edge(START, "researcher")           # 入口点：开始研究
researcher_builder.add_edge("compress_research", END)      # 出口点：压缩完成后结束

# 编译研究员子图，供监督者并发调用
# 注意：researcher_tools 会根据条件返回到 researcher 或 compress_research
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终的综合研究报告，包含 token 限制的重试逻辑。

    该函数获取所有收集的研究发现并使用配置的报告生成模型
    将其综合为结构良好、全面的最终报告。

    工作流程：
    1. 提取研究发现并准备状态清理
    2. 配置最终报告生成模型
    3. 使用 token 限制重试逻辑尝试报告生成
    4. 返回最终报告和清理后的状态

    Args:
        state: 代理状态，包含研究发现和上下文
        config: 运行时配置，包含模型设置和 API 密钥

    Returns:
        包含最终报告和清理后状态的字典
    """
    # ========== 第一步：提取研究发现并准备状态清理 ==========
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # ========== 第二步：配置最终报告生成模型 ==========
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }

    # ========== 第三步：使用 token 限制重试逻辑尝试报告生成 ==========
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # 创建包含所有研究上下文的综合提示词
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            # 返回成功的报告生成结果
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            # ========== 处理 token 限制超出错误 ==========
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # 第一次重试：确定初始截断限制
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"错误：生成最终报告时超出 token 限制，但无法确定模型的最大上下文长度。请在 deep_researcher/utils.py 中的模型映射中更新此信息。{e}",
                            "messages": [AIMessage(content="由于 token 限制，报告生成失败")],
                            **cleared_state
                        }
                    # 使用 4 倍 token 限制作为截断的字符近似值
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每次减少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)

                # 截断发现并重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非 token 限制错误：立即返回错误
                return {
                    "final_report": f"错误：生成最终报告失败：{e}",
                    "messages": [AIMessage(content="由于错误，报告生成失败")],
                    **cleared_state
                }

    # ========== 第四步：如果所有重试都已耗尽，返回失败结果 ==========
    return {
        "final_report": "错误：生成最终报告失败，已达到最大重试次数",
        "messages": [AIMessage(content="经过最大重试次数后，报告生成失败")],
        **cleared_state
    }

# ==================== 主图构建（Deep Researcher Graph）====================
# 这是完整的深度研究工作流，从用户输入到最终报告
# 工作流程：clarify_with_user → write_research_brief → research_supervisor → final_report_generation

# 创建主状态图
# - AgentState: 完整状态类型，包含所有工作流数据
# - AgentInputState: 输入状态类型（仅包含 messages），简化外部调用接口
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,  # 定义输入状态，简化外部调用
    config_schema=Configuration
)

# ========== 添加主工作流节点 ==========
# 每个节点代表工作流中的一个关键阶段
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 用户澄清阶段：判断是否需要澄清问题
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究规划阶段：生成结构化研究简报
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行阶段：嵌入监督者子图，管理研究任务
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成阶段：综合研究发现生成最终报告

# ========== 定义主工作流边 ==========
# START → clarify_with_user：从入口点开始，首先判断是否需要澄清
deep_researcher_builder.add_edge(START, "clarify_with_user")

# research_supervisor → final_report_generation：研究完成后生成报告
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")

# final_report_generation → END：报告生成后结束整个工作流
deep_researcher_builder.add_edge("final_report_generation", END)

# 注意：clarify_with_user 和 write_research_brief 的边由 Command 对象动态控制
# - clarify_with_user 可能返回 END（需要澄清）或 write_research_brief（不需要澄清）
# - write_research_brief 总是返回 research_supervisor

# ========== 编译完整的深度研究工作流 ==========
# 这是对外暴露的主入口点，包含完整的研究流程
deep_researcher = deep_researcher_builder.compile()