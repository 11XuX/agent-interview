"""Synthesis Agent：把证据写成综述。

按子问题分头生成，每个子问题**只喂它自己的证据** —— 模型看不到别的材料，
就编不出没有出处的话。生成完再拼装成一篇。
"""

from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm import MAX_CONCURRENCY, llm
from ..state import State

CITE = "[{paper_id} · {section}]"

chain = ChatPromptTemplate.from_messages([
    ("system",
     "你在写一篇文献综述里的一个小节，回答给定的子问题。\n"
     "只能用给定的证据，一条都不能超出。硬性要求：\n"
     "1. 每个结论后面紧跟它的出处标记，格式 [PMCxxxxx · 章节名]，照抄证据里给的。\n"
     "2. 证据之间有冲突就并列写出来，不要挑一边、也不要调和。\n"
     "3. 只有一条证据支持的结论，要写明「仅一篇文献报告」。\n"
     "4. 不写「综上所述」「总体而言」这类没有出处的概括句。\n"
     "5. 三到六句话，中文，不要小标题。"),
    ("human", "子问题：{question}\n\n可用证据：\n{evidence}{fixes}"),
]) | llm | StrOutputParser()


def _fmt(e) -> str:
    mark = "支持" if e.supports else "反例"
    cite = CITE.format(paper_id=e.paper_id, section=e.section)
    return f"- [{mark}] {cite} {e.claim}\n  原文：{e.quote}"


MIN_KEEP_RATIO = 0.9    # 重写后短于原文这个比例就丢弃重写结果


async def synthesis(state: State) -> State:
    by_q: dict[str, list] = defaultdict(list)
    for e in state.get("evidence", []):
        by_q[e.sub_question].append(e)

    ordered = [sq.question for sq in state["plan"].sub_queries]
    answerable = [q for q in ordered if by_q[q]]

    # 第二轮进来时带着 Review 的意见重写
    issues = state.get("issues") or []
    fixes = ("\n\n上一版被指出的问题，这次必须修掉：\n"
             + "\n".join(f"- {i}" for i in issues)) if issues else ""

    drafts = await chain.abatch(
        [{"question": q, "evidence": "\n".join(_fmt(e) for e in by_q[q]), "fixes": fixes}
         for q in answerable],
        config={"max_concurrency": MAX_CONCURRENCY},
    ) if answerable else []

    parts = [f"# {state['question']}\n"]
    for q, text in zip(answerable, drafts):
        parts.append(f"## {q}\n\n{text.strip()}\n")

    # 没有证据的子问题必须显式说明，不能装作没问过
    missing = [q for q in ordered if not by_q[q]]
    if missing:
        parts.append("## 证据不足，未能回答\n")
        parts += [f"- {q}" for q in missing]
        parts.append("")

    papers = {e.paper_id: e.paper_title for e in state.get("evidence", [])}
    parts.append("## 引用文献\n")
    parts += [f"- `{pid}` {title}" for pid, title in sorted(papers.items())]

    report = "\n".join(parts)

    # 长度守卫：重写有时会越改越短，把内容改没了。照 OpenScholar edit_with_feedback
    # 的做法，短于原文九成就丢弃这次重写。
    old = state.get("report", "")
    if old and len(report) < len(old) * MIN_KEEP_RATIO:
        return {"report": old, "issues": []}

    return {"report": report, "issues": []}
