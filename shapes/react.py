"""形状二：裸 ReAct agent。

图只有两个节点、一条回边，没有一条边编码业务流程。
模型看着工具列表自己决定查什么、查几次、什么时候读全文、什么时候收工。
"""

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from paper_agent.llm import llm

from .tools import TOOLS

SYSTEM = (
    "你是文献调研助手。用工具查资料，最后写一段带出处的综述。\n"
    "出处格式 [PMCxxxxx · 章节名]，只能引用你真的读到过的内容。\n"
    "查不够就继续查，不要急着下结论；材料够了就直接输出综述，不要再调工具。"
)

model = llm.bind_tools(TOOLS)


async def call_model(state: MessagesState) -> dict:
    return {"messages": [await model.ainvoke([("system", SYSTEM)] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)   # ← 模型决定下一步
builder.add_edge("tools", "call_model")                        # ← 工具跑完回模型
graph = builder.compile()
