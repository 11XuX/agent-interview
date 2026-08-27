"""图装配。"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    check, extract, fan_out, planner, ranker, reader,
    review, route_after_check, route_after_review, search, synthesis,
)
from .state import State

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("search", search)
builder.add_node("ranker", ranker)
builder.add_node("check", check)
builder.add_node("reader", reader)
builder.add_node("extract", extract)
builder.add_node("synthesis", synthesis)
builder.add_node("review", review)

builder.add_edge(START, "planner")
builder.add_edge("planner", "search")
builder.add_edge("search", "ranker")
builder.add_edge("ranker", "check")
# 条件边：证据不够就回 search 再查一轮 —— 这条回边是 LCEL 写不出来的
builder.add_conditional_edges("check", route_after_check)
# Send 扇出：每篇论文起一个 extract 实例，全在同一个超步里并发
builder.add_conditional_edges("reader", fan_out, ["extract"])
# 所有 extract 实例跑完才进 synthesis —— 超步天然是同步屏障
builder.add_edge("extract", "synthesis")
builder.add_edge("synthesis", "review")
# 第二条回边：审出问题就回 synthesis 带着意见重写
builder.add_conditional_edges("review", route_after_review)

graph = builder.compile()
