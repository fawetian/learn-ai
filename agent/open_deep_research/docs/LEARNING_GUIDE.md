# Open Deep Research 学习指南

## 推荐的学习顺序

### 第一阶段：理解项目结构

**目标：** 了解项目的整体布局和依赖关系

1. 阅读 `README.md` - 了解项目用途和快速开始
2. 浏览 `langgraph.json` - 了解 LangGraph 配置
3. 浏览 `pyproject.toml` - 了解项目依赖

**关键概念：**
- LangGraph 是什么？它如何编排工作流？
- 项目的入口点在哪里？

### 第二阶段：理解状态和配置

**目标：** 理解数据如何在系统中流动

**阅读顺序：**
1. `src/open_deep_research/state.py` - 理解状态层级和结构化输出
2. `src/open_deep_research/configuration.py` - 理解所有可配置参数

**练习：** 画出状态之间的关系图

**关键概念：**
- `AgentState`、`SupervisorState`、`ResearcherState` 的区别
- `override_reducer` 的作用
- 配置加载优先级

### 第三阶段：理解提示词设计

**目标：** 理解如何通过提示词控制 Agent 行为

**阅读顺序：**
1. `src/open_deep_research/prompts.py` - 理解提示词模板

**重点关注：**
- `lead_researcher_prompt` - 研究主管如何规划
- `research_system_prompt` - 研究员如何执行搜索

**思考：**
- 提示词中的 XML 标签有什么作用？
- 硬性限制（Hard Limits）如何防止无限循环？

### 第四阶段：理解核心工作流

**目标：** 理解完整的研究流程

**阅读顺序：**
1. `src/open_deep_research/deep_researcher.py` - 从底部开始看图的构建

**建议：**
- 先看文件底部的图构建代码
- 理解子图：`supervisor_subgraph`、`researcher_subgraph`
- 逐个理解节点函数（从简单到复杂）

**关键函数阅读顺序：**
1. `clarify_with_user()` - 最简单，理解基本模式
2. `write_research_brief()` - 理解结构化输出
3. `researcher()` 和 `researcher_tools()` - 理解工具调用
4. `compress_research()` - 理解 Token 限制处理
5. `supervisor()` 和 `supervisor_tools()` - 理解并发执行
6. `final_report_generation()` - 理解报告生成

### 第五阶段：理解工具和辅助函数

**目标：** 理解搜索、MCP 和 Token 管理

**阅读顺序：**
1. `src/open_deep_research/utils.py`

**重点关注：**
- `get_all_tools()` - 如何组装工具
- `tavily_search()` 和 `tavily_search_async()` - 搜索实现
- `is_token_limit_exceeded()` - Token 限制处理
- `load_mcp_tools()` - MCP 工具加载

### 第六阶段：实践和扩展

**目标：** 动手实践，加深理解

1. 运行示例研究任务
2. 修改配置参数观察效果
3. 添加自定义 MCP 工具
4. 尝试不同的 LLM 提供商

## 关键概念解释

### LangGraph 核心概念

| 概念 | 说明 |
|------|------|
| **StateGraph** | 状态图，定义节点和边 |
| **Node** | 图中的节点，执行具体逻辑 |
| **Edge** | 节点之间的连接 |
| **Command** | 控制图流转的对象，指定下一个节点和状态更新 |
| **Subgraph** | 嵌套的子图，可以作为节点使用 |

### 状态管理

```python
# 状态定义示例
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 追加模式
    notes: Annotated[list[str], operator.add]            # 追加模式
    final_report: Annotated[str, override_reducer]       # 覆盖模式
```

- `add_messages`: LangChain 提供的消息追加器
- `operator.add`: 列表追加
- `override_reducer`: 自定义覆盖器

### 工具调用模式

```python
# 绑定工具到模型
model_with_tools = model.bind_tools([tool1, tool2, tool3])

# 调用模型获取工具调用
response = await model_with_tools.ainvoke(messages)

# 处理工具调用
for tool_call in response.tool_calls:
    result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
```

### 并发执行

```python
# 创建并发任务
tasks = [
    async_function(arg)
    for arg in args_list
]

# 等待所有任务完成
results = await asyncio.gather(*tasks)
```

## 从哪个文件开始阅读

**推荐起点：** `src/open_deep_research/state.py`

**原因：**
1. 文件较短（299 行），容易理解
2. 定义了所有数据结构，是理解其他文件的基础
3. 展示了 LangGraph 的状态管理模式

**阅读 state.py 时关注：**
1. 结构化输出模型（`ConductResearch`、`ResearchComplete` 等）
2. 状态类型定义（`AgentState`、`SupervisorState`、`ResearcherState`）
3. `override_reducer` 函数的实现

## 学习路径建议

```
state.py (状态定义)
    ↓
configuration.py (配置管理)
    ↓
prompts.py (提示词模板)
    ↓
deep_researcher.py (核心工作流)
    ↓
utils.py (工具函数)
    ↓
实践：运行和修改
```

## 常见问题

### Q: 为什么使用子图？

**A:** 子图提供了以下好处：
1. **封装复杂性** - 将相关逻辑组织在一起
2. **复用** - 子图可以在多个地方使用
3. **独立测试** - 可以单独测试子图
4. **状态隔离** - 子图有自己的状态空间

### Q: Token 限制如何处理？

**A:** 项目使用渐进式截断策略：
1. 检测是否超过 Token 限制
2. 如果超过，移除最早的消息
3. 重试直到成功或达到最大重试次数

### Q: 如何添加新的搜索 API？

**A:**
1. 在 `configuration.py` 中添加新的 `SearchAPI` 枚举值
2. 在 `utils.py` 的 `get_search_tool()` 中添加对应的工具创建逻辑
3. 实现搜索函数

### Q: 如何添加 MCP 工具？

**A:**
1. 配置 MCP 服务器（在 `mcp_config` 中）
2. 工具会自动通过 `load_mcp_tools()` 加载
3. 在提示词中添加工具使用说明（`mcp_prompt`）

## 进阶学习资源

1. **LangGraph 文档** - https://langchain-ai.github.io/langgraph/
2. **LangChain 文档** - https://python.langchain.com/
3. **MCP 规范** - https://modelcontextprotocol.io/
4. **Tavily API** - https://tavily.com/

## 实践项目建议

1. **修改提示词** - 尝试改进研究质量
2. **添加新工具** - 集成其他搜索 API 或数据源
3. **优化并发** - 调整并发参数观察效果
4. **自定义报告格式** - 修改报告生成逻辑
5. **添加缓存** - 缓存搜索结果减少 API 调用
