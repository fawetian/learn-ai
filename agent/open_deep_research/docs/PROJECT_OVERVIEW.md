# Open Deep Research 项目概览

## 项目简介

**Open Deep Research** 是一个完全开源的、可配置的深度研究 Agent，能够自动化执行复杂的研究任务。它支持多个 LLM 提供商、搜索工具和 MCP（Model Context Protocol）服务器，可以并发执行多个研究任务，最终生成结构化的研究报告。

该项目在 Deep Research Bench 排行榜上排名第 6，性能与许多流行的深度研究 Agent 相当。

## 技术栈

| 技术 | 版本/用途 | 说明 |
|------|---------|------|
| **LangGraph** | ≥0.5.4 | 工作流编排和图执行引擎，核心框架 |
| **LangChain** | 最新版 | LLM 集成、工具调用、消息管理 |
| **Pydantic** | 最新版 | 数据验证和配置管理 |
| **Python** | ≥3.10 | 异步编程支持（asyncio） |
| **Tavily** | 默认搜索 API | 高质量网络搜索 |
| **MCP** | ≥1.9.4 | 模型上下文协议，扩展工具能力 |

**支持的 LLM 提供商：**
- OpenAI (GPT-4.1, GPT-5)
- Anthropic (Claude)
- Google (Gemini)
- Groq
- DeepSeek
- Ollama (本地模型)

**支持的搜索 API：**
- Tavily（推荐，独立于模型）
- OpenAI 原生搜索
- Anthropic 原生搜索
- MCP 工具扩展

## 目录结构

```
open_deep_research/
├── src/
│   ├── open_deep_research/          # 核心实现（3120 行代码）
│   │   ├── deep_researcher.py       # 主 LangGraph 工作流 (839 行)
│   │   ├── state.py                 # 状态定义和数据结构 (299 行)
│   │   ├── configuration.py         # 配置管理 (506 行)
│   │   ├── prompts.py               # 系统提示词模板 (480 行)
│   │   └── utils.py                 # 工具函数和辅助方法 (996 行)
│   ├── security/
│   │   └── auth.py                  # 认证处理器
│   └── legacy/                      # 旧版实现（参考用）
│       ├── graph.py                 # 计划-执行工作流
│       └── multi_agent.py           # 多 Agent 架构
├── tests/                           # 测试和评估
│   ├── run_evaluate.py              # 运行评估
│   ├── extract_langsmith_data.py    # 提取评估数据
│   ├── evaluators.py                # 评估器
│   └── expt_results/                # 评估结果
├── examples/                        # 示例输出
│   ├── arxiv.md                     # ArXiv 研究示例
│   ├── pubmed.md                    # PubMed 研究示例
│   └── inference-market.md          # 推理市场研究示例
├── docs/                            # 文档
│   ├── PROJECT_OVERVIEW.md          # 项目概览
│   ├── LEARNING_GUIDE.md            # 学习指南
│   └── FILES.md                     # 文件说明
├── langgraph.json                   # LangGraph 配置
├── pyproject.toml                   # 项目依赖
├── README.md                        # 快速开始指南
└── .env.example                     # 环境变量示例
```

## 核心模块介绍

### 1. deep_researcher.py - 主工作流（839 行）

这是项目的核心入口，定义了完整的 LangGraph 工作流。包含 8 个主要节点函数和 3 个子图：

**主要节点函数：**

| 节点 | 功能 |
|------|------|
| `clarify_with_user()` | 判断用户请求是否清晰，必要时询问澄清问题 |
| `write_research_brief()` | 将用户消息转换为结构化研究简报 |
| `supervisor()` | 研究主管，规划研究策略，分配任务 |
| `supervisor_tools()` | 执行监督工具调用，并发启动子研究员 |
| `researcher()` | 子研究员，执行具体搜索任务 |
| `researcher_tools()` | 处理研究员的工具调用 |
| `compress_research()` | 压缩研究发现，保留关键信息 |
| `final_report_generation()` | 生成最终综合报告 |

### 2. state.py - 状态定义（299 行）

定义了工作流中使用的所有状态类型：

```
AgentState (主图状态)
├── messages: 用户对话历史
├── supervisor_messages: 监督者消息
├── research_brief: 研究简报
├── raw_notes: 原始研究笔记
├── notes: 压缩后的研究笔记
└── final_report: 最终报告

SupervisorState (监督者子图状态)
├── supervisor_messages: 监督者消息
├── research_brief: 研究简报
├── notes: 研究笔记
├── research_iterations: 迭代计数
└── raw_notes: 原始笔记

ResearcherState (研究员子图状态)
├── researcher_messages: 研究员消息
├── tool_call_iterations: 工具调用计数
├── research_topic: 研究主题
├── compressed_research: 压缩结果
└── raw_notes: 原始笔记
```

### 3. configuration.py - 配置管理（506 行）

集中管理所有可配置参数：

| 分类 | 配置项 | 默认值 |
|------|--------|--------|
| **通用配置** | max_structured_output_retries | 3 |
| | allow_clarification | True |
| | max_concurrent_research_units | 5 |
| **研究配置** | search_api | TAVILY |
| | max_researcher_iterations | 10 |
| | max_react_tool_calls | 15 |
| **模型配置** | summarization_model | openai:gpt-4.1-mini |
| | research_model | openai:gpt-4.1 |
| | compression_model | openai:gpt-4.1 |
| | final_report_model | openai:gpt-4.1 |

### 4. prompts.py - 提示词模板（480 行）

包含所有系统提示词，控制 Agent 的行为：

| 提示词 | 用途 |
|--------|------|
| `clarify_with_user_instructions` | 澄清问题指令 |
| `transform_messages_into_research_topic_prompt` | 消息转研究简报 |
| `lead_researcher_prompt` | 研究主管系统提示 |
| `research_system_prompt` | 研究员系统提示 |
| `compress_research_system_prompt` | 研究压缩指令 |
| `final_report_generation_prompt` | 报告生成指令 |

### 5. utils.py - 工具函数（996 行）

提供各种辅助功能：

| 部分 | 功能 |
|------|------|
| **Tavily 搜索** | 执行网络搜索、异步搜索、网页摘要 |
| **MCP 工具** | 加载 MCP 工具、认证包装、获取令牌 |
| **Token 管理** | 检查 token 限制、获取模型限制、消息截断 |
| **工具组装** | 获取所有工具、搜索工具、思考工具 |

## 架构设计

### 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    主图层 (Deep Researcher)                  │
│  clarify_with_user → write_research_brief                   │
│  → research_supervisor → final_report_generation            │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          监督层 (Supervisor Subgraph)                 │   │
│  │  supervisor → supervisor_tools                        │   │
│  │  (规划策略)   (执行工具调用)                          │   │
│  │                      ↓                                │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │    研究层 (Researcher Subgraph × N 并发)       │  │   │
│  │  │  researcher → researcher_tools                 │  │   │
│  │  │  → compress_research                           │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 完整数据流

```
1. 用户输入研究问题
   ↓
2. clarify_with_user 判断是否需要澄清
   ├─ 需要澄清 → 返回澄清问题给用户 → END
   └─ 不需要澄清 ↓
3. write_research_brief 生成结构化研究简报
   ↓
4. research_supervisor (监督者子图)
   ├─ supervisor 规划研究策略
   ├─ supervisor_tools 并发启动多个 researcher
   │  ├─ researcher 执行搜索任务
   │  ├─ researcher_tools 处理搜索结果
   │  └─ compress_research 压缩研究发现
   ├─ 收集所有研究结果
   └─ 循环直到研究完成
   ↓
5. final_report_generation 生成最终报告
   ↓
6. 返回结构化研究报告给用户
```

## 关键设计模式

1. **子图模式** - 将复杂逻辑封装为独立子图，便于复用和测试
2. **工具调用模式** - 使用结构化输出和工具绑定
3. **并发执行模式** - 使用 asyncio.gather 并发执行多个研究任务
4. **重试和容错模式** - 结构化输出重试、Token 限制处理
5. **状态管理模式** - 使用 override_reducer 支持状态覆盖和追加

## 性能指标

根据 Deep Research Bench 评估结果：

| 模型配置 | RACE 分数 | 总成本 | 总 Token |
|---------|---------|--------|----------|
| GPT-5 | 0.4943 | - | 204,640,896 |
| 默认配置 (GPT-4.1) | 0.4309 | $45.98 | 58,015,332 |
| Claude Sonnet 4 | 0.4401 | $187.09 | 138,917,050 |

## 项目亮点

1. **完全开源** - MIT 许可证，代码透明
2. **高度可配置** - 支持多个 LLM、搜索 API、MCP 工具
3. **性能优异** - Deep Research Bench 排名第 6
4. **并发执行** - 支持多个研究员并发工作
5. **容错机制** - Token 限制处理、重试逻辑
6. **易于部署** - 支持多种部署方式
