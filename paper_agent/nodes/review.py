"""Review Agent：查引用真伪 + 查无出处的论断。

两层：

1. 机械校验（纯代码）—— 引用标记是否指向真实存在的证据、有没有整句没出处。
   这层不花钱、不会漏、结论确定，能用代码查的绝不交给模型。
2. 模型自审 —— 论断有没有超出证据支持的范围。这层只做代码查不了的语义判断。

查出问题回 synthesis 重写。照 OpenScholar 的 feedback 环。
"""

import re

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from ..llm import llm
from pydantic import BaseModel, Field

from ..state import State

MAX_REVIEW_ROUNDS = 1        # 最多重写几轮
CITE_RE = re.compile(r"\[([A-Za-z0-9._\-]+)\s*·\s*([^\]]+?)\]")


class Audit(BaseModel):
    """模型自审的结果。"""

    issues: list[str] = Field(
        description="每条一句话，指出哪个论断超出了证据支持的范围。没问题就给空列表"
    )


chain = ChatPromptTemplate.from_messages([
    ("system",
     "你是审稿人。对照证据清单检查综述，只挑一类问题：**论断超出了证据能支持的范围**。\n"
     "比如证据只说了 A 方法在某个数据集上更好，综述写成 A 方法普遍更好；\n"
     "或者证据是单篇报告，综述写成领域共识。\n"
     "引用格式、错别字、行文风格一律不管。没问题就返回空列表，不要凑数。"),
    ("human", "证据清单：\n{evidence}\n\n综述：\n{report}"),
]) | llm.with_structured_output(Audit, method="function_calling")


def _mechanical(report: str, evidence) -> list[str]:
    """代码能查死的部分。"""
    valid = {(e.paper_id, e.section.strip()) for e in evidence}
    issues: list[str] = []

    for pid, sec in CITE_RE.findall(report):
        if (pid, sec.strip()) not in valid:
            issues.append(f"引用 [{pid} · {sec.strip()}] 在证据清单里不存在，删掉或换成真实出处")

    # 正文段落里整段没有任何引用标记的，视为无出处
    for para in report.split("\n\n"):
        para = para.strip()
        if not para or para.startswith(("#", "-", "`")):
            continue
        if not CITE_RE.search(para):
            issues.append(f"这段没有任何出处标记：{para[:50]}...")
    return issues


async def review(state: State) -> State:
    ev = state.get("evidence", [])
    issues = _mechanical(state["report"], ev)

    audit: Audit = await chain.ainvoke({
        "evidence": "\n".join(
            f"- [{e.paper_id} · {e.section}] {e.claim}" for e in ev),
        "report": state["report"],
    })
    issues += audit.issues

    return {"issues": issues, "review_round": state.get("review_round", 0) + 1}


def route_after_review(state: State) -> str:
    """有问题且还没到重写上限就回 synthesis，否则收工。"""
    if state["issues"] and state["review_round"] <= MAX_REVIEW_ROUNDS:
        return "synthesis"
    return END
