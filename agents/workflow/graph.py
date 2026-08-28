"""图装配。

两个变体：

    graph          不带 checkpointer、不带人工确认，用于跑批和对照实验
    with_hitl()    带 checkpointer 和 approve 节点，检索式生成后停下来等人

compile 传不传 checkpointer 决定两件事：能不能断点续跑、能不能用 interrupt。
不传的话 interrupt 会挂起但无处恢复。
"""

from langgraph.graph import END, START, StateGraph

from .models import Evidence, Finding, Paper, Plan, Section, SubQuery
from .nodes import (
    approve,
    check,
    extract,
    fan_out,
    planner,
    ranker,
    reader,
    review,
    route_after_check,
    route_after_review,
    search,
    synthesis,
)
from .state import State


def _build(hitl: bool) -> StateGraph:
    b = StateGraph(State)
    b.add_node("planner", planner)
    b.add_node("search", search)
    b.add_node("ranker", ranker)
    b.add_node("check", check)
    b.add_node("reader", reader)
    b.add_node("extract", extract)
    b.add_node("synthesis", synthesis)
    b.add_node("review", review)

    b.add_edge(START, "planner")
    if hitl:
        b.add_node("approve", approve)
        b.add_edge("planner", "approve")
        b.add_edge("approve", "search")
    else:
        b.add_edge("planner", "search")

    b.add_edge("search", "ranker")
    b.add_edge("ranker", "check")
    # 回边一：证据不够就回 search 再查一轮。LCEL 写不出这个形状
    b.add_conditional_edges("check", route_after_check)
    # Send 扇出：每篇论文起一个 extract 实例，全在同一超步内并发
    b.add_conditional_edges("reader", fan_out, ["extract"])
    # 超步天然是同步屏障：所有 extract 跑完、evidence 合并完才进 synthesis
    b.add_edge("extract", "synthesis")
    b.add_edge("synthesis", "review")
    # 回边二：审出问题就回 synthesis 带着意见重写
    b.add_conditional_edges("review", route_after_review)
    return b


graph = _build(hitl=False).compile()


# 状态里的自定义类必须登记，否则 checkpoint 反序列化时只是警告，未来版本会直接
# 拒绝。这是"状态必须可序列化"的代价 —— 断点续跑与跨进程恢复建立在状态能存能读
# 之上，代价就是状态里不能放任意对象。
CHECKPOINT_TYPES = [Evidence, Finding, Paper, Plan, Section, SubQuery]


def make_serde():
    """带自定义类白名单的序列化器。"""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(
        allowed_msgpack_modules=None,            # 只放行内置安全类型
    ).with_msgpack_allowlist(CHECKPOINT_TYPES)   # 外加本项目这几个


def with_hitl(checkpointer=None):
    """带 checkpointer 和人工确认的版本。

    MemorySaver 存进程内存，进程结束即失效。跨进程恢复需换
    SqliteSaver / PostgresSaver（在独立的包里）。
    """
    from langgraph.checkpoint.memory import MemorySaver

    return _build(hitl=True).compile(
        checkpointer=checkpointer or MemorySaver(serde=make_serde()))
