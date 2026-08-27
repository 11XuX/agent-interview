"""图装配。

compile() 传不传 checkpointer 决定两件事：能不能断点续跑、能不能用 interrupt。
不传的话 interrupt 会直接报错 —— 没地方存档就没法恢复。
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    approve,
    check, extract, fan_out, planner, ranker, reader,
    review, route_after_check, route_after_review, search, synthesis,
)
from .models import Evidence, Finding, Paper, Plan, Section, SubQuery
from .state import State

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("approve", approve)
builder.add_node("search", search)
builder.add_node("ranker", ranker)
builder.add_node("check", check)
builder.add_node("reader", reader)
builder.add_node("extract", extract)
builder.add_node("synthesis", synthesis)
builder.add_node("review", review)

builder.add_edge(START, "planner")
builder.add_edge("planner", "approve")
builder.add_edge("approve", "search")
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

# 不带 checkpointer 的版本：跑批、跑分用，没有中断
graph = builder.compile()


# 状态里的自定义类必须登记，否则 checkpoint 反序列化时只是警告，
# 未来版本会直接拒绝。这是"状态必须可序列化"的代价 —— LangGraph 的
# 断点续跑、跨进程恢复全建立在状态能存能读之上，代价就是状态里不能
# 塞任意对象。
CHECKPOINT_TYPES = [Evidence, Finding, Paper, Plan, Section, SubQuery]


def make_serde():
    """带自定义类白名单的序列化器。"""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(
        allowed_msgpack_modules=None,           # 只放行内置安全类型
    ).with_msgpack_allowlist(CHECKPOINT_TYPES)  # 外加我们自己这几个


def with_hitl(checkpointer=None):
    """带 checkpointer 的版本，approve 节点会真的挂起等人。

    MemorySaver 存在进程内存里，进程一死存档就没了。要跨进程恢复
    换 SqliteSaver / PostgresSaver（在 langgraph-checkpoint-sqlite
    等单独的包里）。
    """
    from langgraph.checkpoint.memory import MemorySaver

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(serde=make_serde()))
