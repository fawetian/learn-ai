"""
@file configuration.py
@description 配置管理模块，集中管理 Deep Research Agent 的所有可配置参数

主要功能：
- 定义搜索 API 枚举（SearchAPI）
- 定义 MCP 服务器配置（MCPConfig）
- 定义主配置类（Configuration），包含模型、搜索、并发等设置
- 支持从环境变量和 RunnableConfig 加载配置

依赖关系：
- pydantic.BaseModel: 配置验证
- langchain_core.runnables.RunnableConfig: 运行时配置

配置优先级（从高到低）：
1. RunnableConfig 中的 configurable 参数
2. 环境变量（字段名大写）
3. 默认值

配置分类：
┌─────────────────────────────────────────────────────────────┐
│                    Configuration                             │
├─────────────────────────────────────────────────────────────┤
│ 通用配置                                                     │
│   ├── max_structured_output_retries: 结构化输出重试次数      │
│   ├── allow_clarification: 是否允许向用户询问澄清问题        │
│   └── max_concurrent_research_units: 最大并发研究单元数      │
├─────────────────────────────────────────────────────────────┤
│ 研究配置                                                     │
│   ├── search_api: 搜索 API 选择                             │
│   ├── max_researcher_iterations: 最大研究迭代次数            │
│   └── max_react_tool_calls: 单次研究的最大工具调用次数       │
├─────────────────────────────────────────────────────────────┤
│ 模型配置                                                     │
│   ├── summarization_model: 摘要模型                         │
│   ├── research_model: 研究模型                              │
│   ├── compression_model: 压缩模型                           │
│   └── final_report_model: 报告生成模型                      │
├─────────────────────────────────────────────────────────────┤
│ MCP 配置                                                     │
│   ├── mcp_config: MCP 服务器配置                            │
│   └── mcp_prompt: MCP 工具使用说明                          │
└─────────────────────────────────────────────────────────────┘
"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """
    搜索 API 提供商枚举

    定义了系统支持的所有搜索 API 选项。不同的搜索 API 有不同的特点：
    - ANTHROPIC: Anthropic 原生网络搜索，需要 Claude 模型支持
    - OPENAI: OpenAI 原生网络搜索，需要 GPT 模型支持
    - TAVILY: 第三方搜索 API，独立于模型，推荐用于大多数场景
    - NONE: 不使用搜索，仅依赖 MCP 工具或模型知识

    注意：选择搜索 API 时需要确保研究模型支持该 API
    """
    ANTHROPIC = "anthropic"  # Anthropic 原生搜索（需要 Claude 模型）
    OPENAI = "openai"        # OpenAI 原生搜索（需要 GPT 模型）
    TAVILY = "tavily"        # Tavily 搜索 API（推荐，独立于模型）
    NONE = "none"            # 不使用搜索


class MCPConfig(BaseModel):
    """
    MCP（Model Context Protocol）服务器配置

    MCP 是一种协议，允许 LLM 与外部工具和服务进行交互。
    通过配置 MCP 服务器，可以扩展 Agent 的能力，例如：
    - 访问本地文件系统
    - 查询数据库
    - 调用自定义 API

    Attributes:
        url: MCP 服务器的 URL 地址
        tools: 要暴露给 LLM 的工具列表（如果为 None，则暴露所有工具）
        auth_required: 是否需要认证（如果为 True，需要配置认证信息）

    Example:
        MCPConfig(
            url="http://localhost:8080",
            tools=["read_file", "write_file"],
            auth_required=False
        )
    """
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """MCP 服务器的 URL 地址"""

    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """要暴露给 LLM 的工具列表，None 表示暴露所有工具"""

    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """是否需要认证访问 MCP 服务器"""


class Configuration(BaseModel):
    """
    Deep Research Agent 的主配置类

    集中管理所有可配置参数，支持从环境变量和 RunnableConfig 加载。
    配置项通过 metadata 中的 x_oap_ui_config 定义 UI 展示方式。

    配置加载优先级：
    1. RunnableConfig.configurable 中的值
    2. 环境变量（字段名大写，如 SEARCH_API）
    3. 字段默认值

    Attributes:
        # 通用配置
        max_structured_output_retries: 结构化输出失败时的最大重试次数
        allow_clarification: 是否允许 Agent 向用户询问澄清问题
        max_concurrent_research_units: 最大并发研究单元数

        # 研究配置
        search_api: 使用的搜索 API
        max_researcher_iterations: 研究主管的最大迭代次数
        max_react_tool_calls: 单个研究员的最大工具调用次数

        # 模型配置
        summarization_model: 用于摘要网页内容的模型
        research_model: 用于执行研究的模型
        compression_model: 用于压缩研究结果的模型
        final_report_model: 用于生成最终报告的模型

        # MCP 配置
        mcp_config: MCP 服务器配置
        mcp_prompt: MCP 工具的额外使用说明
    """

    # ==================== 通用配置 ====================

    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "模型结构化输出失败时的最大重试次数"
            }
        }
    )
    """
    结构化输出的最大重试次数

    当 LLM 返回的结构化输出不符合预期格式时，会自动重试。
    增加此值可以提高成功率，但会增加延迟和成本。
    """

    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否允许研究员在开始研究前向用户询问澄清问题"
            }
        }
    )
    """
    是否允许向用户询问澄清问题

    如果为 True，当用户请求不够清晰时，Agent 会先询问澄清问题。
    如果为 False，Agent 会直接开始研究，可能导致结果不符合预期。
    """

    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "最大并发研究单元数。允许研究员使用多个子图进行并发研究。注意：并发数越多，越容易触发 API 速率限制。"
            }
        }
    )
    """
    最大并发研究单元数

    控制同时运行的研究员子图数量。增加此值可以加快研究速度，
    但可能会触发 API 速率限制。建议根据 API 配额调整。
    """

    # ==================== 研究配置 ====================

    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "用于研究的搜索 API。注意：确保研究模型支持所选的搜索 API。",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI 原生网络搜索", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic 原生网络搜索", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "无", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    """
    搜索 API 选择

    - tavily: 推荐选项，独立于模型，支持所有 LLM
    - openai: OpenAI 原生搜索，仅支持 GPT 模型
    - anthropic: Anthropic 原生搜索，仅支持 Claude 模型
    - none: 不使用搜索，依赖 MCP 工具或模型知识
    """

    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "研究主管的最大迭代次数。即研究主管反思研究结果并提出后续问题的次数。"
            }
        }
    )
    """
    研究主管的最大迭代次数

    控制 Supervisor 可以进行多少轮研究。每轮研究包括：
    1. 评估当前收集的信息
    2. 决定是否需要更多研究
    3. 分配新的研究任务

    增加此值可以获得更深入的研究，但会增加时间和成本。
    """

    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "单个研究员在一次研究步骤中的最大工具调用次数。"
            }
        }
    )
    """
    单个研究员的最大工具调用次数

    限制每个研究员子图可以进行的搜索次数。
    防止研究员陷入无限循环或过度搜索。
    """

    # ==================== 模型配置 ====================

    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "用于摘要 Tavily 搜索结果中网页内容的模型"
            }
        }
    )
    """
    摘要模型

    用于将搜索结果中的网页内容压缩为简洁摘要。
    推荐使用较小的模型以降低成本，如 gpt-4.1-mini。
    格式：provider:model_name
    """

    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "摘要模型的最大输出 token 数"
            }
        }
    )
    """摘要模型的最大输出 token 数"""

    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "摘要前网页内容的最大字符长度"
            }
        }
    )
    """
    网页内容的最大字符长度

    超过此长度的内容会被截断后再进行摘要。
    用于控制发送给摘要模型的内容大小。
    """

    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "用于执行研究的模型。注意：确保研究模型支持所选的搜索 API。"
            }
        }
    )
    """
    研究模型

    用于执行具体研究任务的模型，包括：
    - 生成搜索查询
    - 分析搜索结果
    - 决定是否需要更多搜索

    注意：如果使用 OpenAI/Anthropic 原生搜索，需要选择对应的模型。
    """

    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "研究模型的最大输出 token 数"
            }
        }
    )
    """研究模型的最大输出 token 数"""

    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "用于压缩子图研究结果的模型。注意：确保压缩模型支持所选的搜索 API。"
            }
        }
    )
    """
    压缩模型

    用于将研究员收集的原始信息压缩为结构化笔记。
    压缩后的笔记会传递给 Supervisor 进行评估。
    """

    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "压缩模型的最大输出 token 数"
            }
        }
    )
    """压缩模型的最大输出 token 数"""

    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "用于根据所有研究结果生成最终报告的模型"
            }
        }
    )
    """
    最终报告模型

    用于根据所有研究笔记生成最终的研究报告。
    推荐使用能力较强的模型以确保报告质量。
    """

    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "最终报告模型的最大输出 token 数"
            }
        }
    )
    """最终报告模型的最大输出 token 数"""

    # ==================== MCP 配置 ====================

    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP 服务器配置"
            }
        }
    )
    """
    MCP 服务器配置

    配置后，Agent 可以使用 MCP 服务器提供的工具。
    这允许 Agent 访问本地文件、数据库等外部资源。
    """

    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "关于 MCP 工具使用的额外指令，传递给 Agent"
            }
        }
    )
    """
    MCP 工具使用说明

    额外的指令，告诉 Agent 如何使用 MCP 工具。
    例如："使用 read_file 工具读取本地文档，优先使用本地资源"
    """

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        从 RunnableConfig 创建配置实例

        这是创建 Configuration 实例的推荐方式，它会自动处理：
        1. 从 RunnableConfig.configurable 读取配置
        2. 从环境变量读取配置（字段名大写）
        3. 使用默认值填充未指定的配置

        Args:
            config: LangChain 的 RunnableConfig，通常由图执行时传入

        Returns:
            Configuration 实例

        Example:
            # 在图节点中使用
            async def my_node(state, config: RunnableConfig):
                cfg = Configuration.from_runnable_config(config)
                model = cfg.research_model
                ...

            # 直接创建（使用环境变量和默认值）
            cfg = Configuration.from_runnable_config()
        """
        # ========== 第一步：获取 configurable 字典 ==========
        # 如果 config 存在，从中提取 configurable；否则使用空字典
        configurable = config.get("configurable", {}) if config else {}

        # ========== 第二步：获取所有字段名 ==========
        field_names = list(cls.model_fields.keys())

        # ========== 第三步：构建配置值字典 ==========
        # 优先级：configurable > 环境变量 > 默认值
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }

        # ========== 第四步：创建实例 ==========
        # 过滤掉 None 值，让 Pydantic 使用默认值
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """
        Pydantic 模型配置

        arbitrary_types_allowed: 允许使用任意类型（如 Enum）
        """
        arbitrary_types_allowed = True
