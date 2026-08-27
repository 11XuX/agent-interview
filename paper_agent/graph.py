"""图装配。"""

from langgraph.graph import END, START, StateGraph

from .nodes import planner, search
from .state import State

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("search", search)

builder.add_edge(START, "planner")
builder.add_edge("planner", "search")
builder.add_edge("search", END)

graph = builder.compile()
