"""
@file state.py
@description 图状态定义和数据结构，用于 Deep Research Agent

主要功能：
- 定义结构化输出模型（ConductResearch, ResearchComplete, Summary 等）
- 定义各层级的状态类型（AgentState, SupervisorState, ResearcherState）
- 提供状态归约器（reducer）用于状态更新

依赖关系：
- langgraph.graph.MessagesState: 消息状态基类
- pydantic.BaseModel: 数据验证基类

状态层级关系：
    AgentState (主图状态)
        ├── supervisor_messages: 监督者的消息历史
        ├── research_brief: 研究简报
        ├── raw_notes: 原始研究笔记
        ├── notes: 压缩后的研究笔记
        └── final_report: 最终报告
            │
            ▼
    SupervisorState (监督者子图状态)
        ├── supervisor_messages: 监督者消息
        ├── research_brief: 研究简报
        ├── notes: 研究笔记
        └── research_iterations: 研究迭代次数
            │
            ▼
    ResearcherState (研究员子图状态)
        ├── researcher_messages: 研究员消息
        ├── tool_call_iterations: 工具调用次数
        ├── research_topic: 研究主题
        └── compressed_research: 压缩后的研究结果
"""

# 导入必要的库
import operator  # 用于状态归约操作（如 operator.add）
from typing import Annotated, Optional  # 类型注解相关

# LangChain 和 LangGraph 核心库
from langchain_core.messages import MessageLikeRepresentation  # 消息类型定义
from langgraph.graph import MessagesState  # 消息状态基类
from pydantic import BaseModel, Field  # 数据验证和字段定义
from typing_extensions import TypedDict  # 类型字典定义


###################
# Structured Outputs（结构化输出模型）
# 这些模型用于 LLM 的工具调用，定义了 Agent 可以执行的操作
###################

class ConductResearch(BaseModel):
    """
    研究任务委派工具

    当 Supervisor 需要委派研究任务给子研究员时调用此工具。
    每次调用会启动一个新的研究员子图来执行具体的搜索任务。

    Attributes:
        research_topic: 研究主题，应该是详细的描述（至少一段话），
                       以便研究员能够准确理解需要搜索的内容

    Example:
        ConductResearch(
            research_topic="深入研究 LangGraph 的状态管理机制，包括状态定义、
                          状态更新、以及如何在节点之间传递状态"
        )
    """
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


class ResearchComplete(BaseModel):
    """
    研究完成标记工具

    当 Supervisor 判断已收集到足够的信息，可以生成最终报告时调用此工具。
    调用后会触发最终报告生成流程。

    注意：这是一个无参数的工具，仅作为信号使用
    """
    pass


class Summary(BaseModel):
    """
    研究摘要模型

    用于存储网页内容的摘要和关键摘录，由 summarize_webpage 函数生成。

    Attributes:
        summary: 内容的简洁摘要
        key_excerpts: 关键引用或摘录，用于支持摘要中的观点
    """
    summary: str
    key_excerpts: str


class ClarifyWithUser(BaseModel):
    """
    用户澄清请求模型

    当用户的研究请求不够清晰时，Agent 使用此模型来决定是否需要
    向用户询问澄清问题。

    Attributes:
        need_clarification: 是否需要向用户询问澄清问题
        question: 要问用户的具体问题（如果 need_clarification 为 True）
        verification: 确认消息，告知用户我们将在获得必要信息后开始研究

    Example:
        ClarifyWithUser(
            need_clarification=True,
            question="您希望报告侧重于技术实现细节还是高层概念介绍？",
            verification="收到您的回复后，我将立即开始研究。"
        )
    """
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    """
    研究问题模型

    用于将用户的原始消息转换为结构化的研究简报。
    研究简报会指导后续的研究方向和范围。

    Attributes:
        research_brief: 研究简报，包含研究问题、范围、预期输出等
    """
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions（状态定义）
# 这些类定义了图中各节点之间传递的状态结构
###################

def override_reducer(current_value, new_value):
    """
    自定义状态归约器，支持覆盖和追加两种模式

    LangGraph 默认使用追加模式更新列表状态，但有时我们需要完全替换状态值。
    此归约器通过检查特殊的 "override" 标记来决定使用哪种模式。

    Args:
        current_value: 当前状态值
        new_value: 新的状态值，可以是普通值或带有 override 标记的字典

    Returns:
        更新后的状态值

    Example:
        # 追加模式（默认）
        override_reducer(["a", "b"], ["c"])  # 返回 ["a", "b", "c"]

        # 覆盖模式
        override_reducer(["a", "b"], {"type": "override", "value": ["x"]})  # 返回 ["x"]
    """
    # 检查是否是覆盖模式
    # 如果 new_value 是字典且包含 type="override"，则完全替换当前值
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        # 否则使用默认的追加模式
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """
    Agent 输入状态

    仅包含 messages 字段，用于接收用户输入。
    继承自 MessagesState，自动处理消息的追加和管理。
    """
    pass


class AgentState(MessagesState):
    """
    主 Agent 状态，用于 Deep Researcher 主图

    这是整个研究流程的顶层状态，包含了从用户输入到最终报告的所有数据。

    Attributes:
        messages: 继承自 MessagesState，存储与用户的对话历史
        supervisor_messages: 监督者的内部消息历史，用于跟踪研究规划
        research_brief: 从用户消息转换而来的结构化研究简报
        raw_notes: 原始研究笔记，直接从搜索结果中提取
        notes: 压缩后的研究笔记，经过整理和去重
        final_report: 最终生成的研究报告

    状态流转：
        1. 用户输入 → messages
        2. 澄清/转换 → research_brief
        3. 研究执行 → raw_notes → notes
        4. 报告生成 → final_report
    """
    # 监督者消息使用 override_reducer，允许在需要时重置消息历史
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]

    # 研究简报，由 write_research_brief 节点生成
    research_brief: Optional[str]

    # 原始研究笔记，每个研究员的搜索结果都会追加到这里
    raw_notes: Annotated[list[str], override_reducer] = []

    # 压缩后的研究笔记，经过 compress_research 处理
    notes: Annotated[list[str], override_reducer] = []

    # 最终报告，由 final_report_generation 节点生成
    final_report: str


class SupervisorState(TypedDict):
    """
    监督者子图状态

    用于 Supervisor Subgraph，管理研究任务的分配和协调。

    Attributes:
        supervisor_messages: 监督者的消息历史，包含规划和决策过程
        research_brief: 研究简报，指导研究方向
        notes: 已收集的研究笔记
        research_iterations: 研究迭代次数，用于控制研究深度
        raw_notes: 原始研究笔记

    工作流程：
        1. 接收 research_brief
        2. 规划研究策略，调用 ConductResearch
        3. 收集研究结果到 notes
        4. 判断是否需要更多研究（基于 research_iterations）
        5. 调用 ResearchComplete 结束研究
    """
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0  # 跟踪研究迭代次数，防止无限循环
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherState(TypedDict):
    """
    研究员子图状态

    用于 Researcher Subgraph，执行具体的搜索和信息收集任务。

    Attributes:
        researcher_messages: 研究员的消息历史，包含搜索查询和结果
        tool_call_iterations: 工具调用次数，用于限制搜索深度
        research_topic: 当前研究的主题（由 Supervisor 分配）
        compressed_research: 压缩后的研究结果
        raw_notes: 原始搜索结果

    工作流程：
        1. 接收 research_topic
        2. 执行搜索工具调用
        3. 收集结果到 raw_notes
        4. 压缩结果到 compressed_research
        5. 返回给 Supervisor
    """
    # 研究员消息使用标准的 operator.add，始终追加
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]

    # 工具调用计数器，防止过多的搜索请求
    tool_call_iterations: int = 0

    # 研究主题，由 Supervisor 通过 ConductResearch 工具指定
    research_topic: str

    # 压缩后的研究结果，由 compress_research 节点生成
    compressed_research: str

    # 原始搜索结果
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    """
    研究员输出状态

    定义研究员子图返回给监督者的数据结构。

    Attributes:
        compressed_research: 压缩后的研究结果摘要
        raw_notes: 原始研究笔记列表
    """
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
