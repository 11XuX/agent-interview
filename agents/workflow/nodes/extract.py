"""Evidence Store：从每篇论文的正文里抽出可追溯的证据。

这是全图唯一用 Send 的地方。Send 是 LangGraph 的扇出原语：
条件边返回一个 Send 列表，框架就在**同一个超步**里并发起 N 个节点实例，
每个实例拿到的"状态"是 Send 自己带的载荷，不是整张图的状态。

N 个实例同时往 evidence 里写，所以 evidence 必须有 reducer。
"""

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send

from ..llm import llm
from ..models import Evidence, Findings, Paper
from ..state import State

MAX_QUOTE = 400     # 单条引文上限，防止模型整段复制

chain = ChatPromptTemplate.from_messages([
    ("system",
     "你是文献证据抽取员。从给定正文里找出能直接回答子问题的原文片段。\n"
     "硬性要求：\n"
     "1. quote 必须逐字来自正文，一个字都不能改，也不能拼接不相邻的句子。\n"
     "2. section 必须是给定的节标题之一，原样抄。\n"
     "3. sub_question 必须是给定子问题之一，原样抄。\n"
     "4. 正文里没有直接证据就返回空列表。不许用常识补，不许写正文没说的话。\n"
     "5. 反例也要抽，用 supports=false 标出来 —— 否定证据和支持证据一样有价值。"),
    ("human", "子问题：\n{questions}\n\n论文：{title}\n\n正文：\n{body}"),
]) | llm.with_structured_output(Findings, method="function_calling")


def fan_out(state: State) -> list[Send]:
    """条件边返回 Send 列表 —— 每篇论文起一个 extract 实例。"""
    return [
        Send("extract", {"paper": p, "questions": p.found_for})
        for p in state["papers"]
    ]


async def extract(payload: dict) -> State:
    """注意入参不是 State，是 Send 带过来的载荷。"""
    p: Paper = payload["paper"]
    questions: list[str] = payload["questions"]

    # 有全文用全文，只有摘要就用摘要 —— arXiv 那几篇走这条路
    if p.sections:
        body = "\n\n".join(f"## {s.title}\n{s.text}" for s in p.sections)
        sections = [s.title for s in p.sections]
    else:
        body = f"## Abstract\n{p.abstract}"
        sections = ["Abstract"]

    if not body.strip() or not questions:
        return {"evidence": []}

    result: Findings = await chain.ainvoke({
        "questions": "\n".join(f"- {q}" for q in questions),
        "title": p.title,
        "body": body,
    })

    out: list[Evidence] = []
    for f in result.findings:
        # 模型可能把 section / sub_question 写歪，对不上的直接丢，不猜
        if not f.quote or f.section not in sections or f.sub_question not in questions:
            continue
        out.append(Evidence(
            **f.model_dump(exclude={"quote"}),
            quote=f.quote[:MAX_QUOTE],
            paper_id=p.ext_id,
            paper_title=p.title,
        ))
    return {"evidence": out}
