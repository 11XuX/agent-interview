"""三形状对照。同一个问题、同一个模型、同一套工具，量四类指标。

指标选择的理由：

    成本 —— token 数是唯一能横向比的成本口径，墙钟受网络波动影响大
    步数 —— 工具调用次数直接反映"模型自己编排"要付的探索代价
    可追溯 —— 引用标记能不能对上真实读到过的内容，是这个业务的核心质量
    覆盖 —— 报告有没有回答到问题的各个方面

引用真伪的判定方式不同：workflow 版有 evidence 清单可以逐条比对；
另外两版没有清单，只能核对 PMCID 是否在工具返回里出现过。
"""

import asyncio
import re
import time
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler

QUESTION = "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"
CITE_RE = re.compile(r"\[(PMC\d+)\s*·\s*([^\]]+?)\]")
PMCID_RE = re.compile(r"PMC\d+")


class Meter(BaseCallbackHandler):
    """挂在 config 上统计 token 和模型调用次数。"""

    def __init__(self):
        self.in_tok = self.out_tok = self.calls = 0

    def on_llm_end(self, response, **kw):
        self.calls += 1
        for gens in response.generations:
            for g in gens:
                u = getattr(g.message, "usage_metadata", None) or {}
                self.in_tok += u.get("input_tokens", 0)
                self.out_tok += u.get("output_tokens", 0)


@dataclass
class Result:
    shape: str
    seconds: float = 0.0
    llm_calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    tool_calls: int = 0
    report: str = ""
    seen_pmcids: set = field(default_factory=set)   # 工具真的返回过的 PMCID

    @property
    def cites(self) -> list[tuple[str, str]]:
        return CITE_RE.findall(self.report)

    @property
    def fake_cites(self) -> list[str]:
        """引用了但工具从没返回过这个 PMCID —— 一定是编的。"""
        return sorted({p for p, _ in self.cites if p not in self.seen_pmcids})


async def _run_messages_shape(name, graph, limit: int) -> Result:
    """react / harness 两版都是 messages 形状，跑法一样。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    r, meter = Result(name), Meter()
    t = time.perf_counter()
    final = ""
    async for chunk in graph.astream(
        {"messages": [HumanMessage(QUESTION)]},
        {"recursion_limit": limit, "callbacks": [meter]},
    ):
        for _node, patch in (chunk or {}).items():
            for m in (patch or {}).get("messages", []) or []:
                if isinstance(m, AIMessage):
                    r.tool_calls += len(m.tool_calls or [])
                    if m.content and not m.tool_calls:
                        final = m.content if isinstance(m.content, str) else str(m.content)
                elif isinstance(m, ToolMessage):
                    r.seen_pmcids |= set(PMCID_RE.findall(str(m.content)))
    r.seconds = time.perf_counter() - t
    r.llm_calls, r.in_tok, r.out_tok, r.report = meter.calls, meter.in_tok, meter.out_tok, final
    return r


async def run_workflow() -> Result:
    from paper_agent.graph import graph

    r, meter = Result("workflow"), Meter()
    t = time.perf_counter()
    out = await graph.ainvoke({"question": QUESTION}, {"callbacks": [meter]})
    r.seconds = time.perf_counter() - t
    r.llm_calls, r.in_tok, r.out_tok = meter.calls, meter.in_tok, meter.out_tok
    r.report = out.get("report", "")
    # workflow 版没有"工具调用"概念，用 HTTP 检索次数近似：每轮每源每检索式一次
    r.tool_calls = len(out.get("tried", [])) * 2 + len(out.get("papers", []))
    r.seen_pmcids = {p.ext_id for p in out.get("papers", [])}
    return r


async def run_react() -> Result:
    from .react import graph
    return await _run_messages_shape("react", graph, 40)


async def run_harness() -> Result:
    from .harness import graph
    return await _run_messages_shape("harness", graph, 60)


def table(rows: list[Result]) -> str:
    head = f"{'':10} {'秒':>6} {'模型调用':>8} {'输入tok':>9} {'输出tok':>9} {'工具次':>7} {'报告字':>7} {'引用':>5} {'伪引用':>6}"
    lines = [head, "─" * len(head)]
    for r in rows:
        lines.append(
            f"{r.shape:10} {r.seconds:6.0f} {r.llm_calls:8d} {r.in_tok:9d} {r.out_tok:9d} "
            f"{r.tool_calls:7d} {len(r.report):7d} {len(r.cites):5d} {len(r.fake_cites):6d}")
    return "\n".join(lines)


async def main():
    rows = []
    for label, fn in (("workflow", run_workflow), ("react", run_react), ("harness", run_harness)):
        print(f"跑 {label} ...", flush=True)
        try:
            rows.append(await fn())
        except Exception as e:                      # noqa: BLE001
            print(f"  {label} 挂了: {type(e).__name__}: {str(e)[:120]}")
    print("\n" + table(rows))
    for r in rows:
        if r.fake_cites:
            print(f"\n{r.shape} 的伪引用: {r.fake_cites}")
    return rows


if __name__ == "__main__":
    asyncio.run(main())
