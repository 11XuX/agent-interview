"""图装配。"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    check, extract, fan_out, planner, ranker, reader,
    route_after_check, search,
)
from .state import State

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("search", search)
builder.add_node("ranker", ranker)
builder.add_node("check", check)
builder.add_node("reader", reader)
builder.add_node("extract", extract)

builder.add_edge(START, "planner")
builder.add_edge("planner", "search")
builder.add_edge("search", "ranker")
builder.add_edge("ranker", "check")
# 条件边：证据不够就回 search 再查一轮 —— 这条回边是 LCEL 写不出来的
builder.add_conditional_edges("check", route_after_check)
# Send 扇出：每篇论文起一个 extract 实例，全在同一个超步里并发
builder.add_conditional_edges("reader", fan_out, ["extract"])
builder.add_edge("extract", END)

graph = builder.compile()
