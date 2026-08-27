"""图装配。"""

from langgraph.graph import END, START, StateGraph

from .nodes import check, planner, ranker, route_after_check, search
from .state import State

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("search", search)
builder.add_node("ranker", ranker)
builder.add_node("check", check)

builder.add_edge(START, "planner")
builder.add_edge("planner", "search")
builder.add_edge("search", "ranker")
builder.add_edge("ranker", "check")
# 条件边：证据不够就回 search 再查一轮 —— 这条回边是 LCEL 写不出来的
builder.add_conditional_edges("check", route_after_check)

graph = builder.compile()
