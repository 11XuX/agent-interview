from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    count: int
    log: Annotated[list[str], list.__add__]


def add_one(state: State) -> dict:
    return {"count": state["count"] + 1, "log": ["add_one 跑了"]}


def double(state: State) -> dict:
    return {"count": state["count"] * 2, "log": ["double 跑了"]}


def route(state: State) -> str:
    """条件边的判断函数：读状态，返回下一个节点的名字。

    它不是节点 —— 不返回补丁，不改状态，只回答"接下来跑谁"。
    """
    return "double" if state["count"] % 2 == 0 else END


builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_node("double", double)
builder.add_edge(START, "add_one")
builder.add_conditional_edges("add_one", route)   # ← 替换掉原来写死的 add_edge
builder.add_edge("double", END)
graph = builder.compile()

for start in (0, 1):
    out = graph.invoke({"count": start, "log": []})
    print(f"  起点 {start} -> count={out['count']:2d}  路径 {out['log']}")
