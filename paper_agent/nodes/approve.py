"""人工确认检索式。

位置选在 planner 之后：检索式写歪了后面全白跑，人扫一眼两秒钟的事。

**必须单独起一个节点，不能把 interrupt 塞进 planner 里面。**
原因是 interrupt 的恢复语义：resume 时被中断的那个节点**从头重新执行**，
不是从 interrupt 那一行继续。塞进 planner 就等于每次恢复都重跑一次 LLM，
既费钱，temperature 不为 0 时还会拿到和人刚看过的不一样的检索式。

所以规矩是：interrupt 所在的节点必须是纯的、可重复执行的。
"""

from langgraph.types import interrupt

from ..models import SubQuery
from ..state import State


def approve(state: State) -> State:
    """把检索式摆给人看。返回值有三种形态：

        None / "ok"     放行
        "skip"          跳过确认（自动化跑批时用）
        [{...}, ...]    人改过的检索式，直接替换
    """
    answer = interrupt({
        "问题": state["question"],
        "待确认的检索式": [
            {"子问题": sq.question, "检索式": sq.query} for sq in state["pending"]
        ],
        "怎么回": "回 ok 放行；回一个 [{question, query}] 列表则替换",
    })

    if isinstance(answer, list) and answer:
        return {"pending": [SubQuery(**a) for a in answer],
                "plan": state["plan"].model_copy(
                    update={"sub_queries": [SubQuery(**a) for a in answer]})}
    return {}
