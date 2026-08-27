"""同一个业务的 agent 形状。对照 paper_agent/ 那个 workflow 形状。

区别只有一处：这里没有一条边是我写的业务流程。模型看着工具列表
自己决定查什么、查几次、什么时候读全文、什么时候收工。
"""

import asyncio
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from paper_agent.llm import llm

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


@tool
async def search_papers(query: str) -> str:
    """按检索式查 Europe PMC。检索式用英文，核心概念 AND，同义词 OR 括起来。

    返回每篇的 PMCID、年份、标题、摘要前 300 字。
    """
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{EPMC}/search", params={
            "query": f"({query}) AND OPEN_ACCESS:Y", "format": "json",
            "pageSize": 5, "resultType": "core"})
    out = []
    for it in r.json().get("resultList", {}).get("result", []):
        out.append(f"{it.get('pmcid', '?')} ({it.get('pubYear')}) {it.get('title', '')[:80]}\n"
                   f"  {(it.get('abstractText') or '无摘要')[:300]}")
    return "\n".join(out) or "没查到"


@tool
async def read_section(pmcid: str, section: str) -> str:
    """读某篇论文的某一节全文。section 传 Methods / Results / Discussion 等。

    只在摘要不足以判断时才用 —— 一次调用会返回几千字。
    """
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.get(f"{EPMC}/{pmcid}/fullTextXML")
    if r.status_code != 200:
        return f"{pmcid} 拿不到全文"
    root = ET.fromstring(r.text)
    body = root.find(".//body")
    for sec in (body.findall("sec") if body is not None else []):
        title = (sec.findtext("title") or "").strip()
        if section.lower() in title.lower():
            return f"[{pmcid} · {title}]\n" + " ".join("".join(sec.itertext()).split())[:2500]
    return f"{pmcid} 没有叫 {section} 的章节"


TOOLS = [search_papers, read_section]
model = llm.bind_tools(TOOLS)

SYSTEM = (
    "你是文献调研助手。用工具查资料，最后写一段带出处的综述。\n"
    "出处格式 [PMCxxxxx · 章节名]，只能引用你真的读到过的内容。\n"
    "查不够就继续查，不要急着下结论；材料够了就直接输出综述，不要再调工具。"
)


async def call_model(state: MessagesState) -> dict:
    return {"messages": [await model.ainvoke(
        [("system", SYSTEM)] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)   # ← 模型决定
builder.add_edge("tools", "call_model")                        # ← 工具跑完回模型
graph = builder.compile()


if __name__ == "__main__":
    async def main():
        msgs = [("user", "单细胞 RNA-seq 的批次效应校正方法哪类更可靠")]
        async for chunk in graph.astream({"messages": msgs},
                                         {"recursion_limit": 40}):
            for node, patch in chunk.items():
                for m in patch["messages"]:
                    if getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            args = str(tc["args"])[:78]
                            print(f"  模型决定调用 → {tc['name']}({args})")
                    elif m.__class__.__name__ == "ToolMessage":
                        print(f"    工具返回 {len(m.content)} 字")
                    elif m.content:
                        print(f"\n── 模型最终输出 ──\n{m.content[:1400]}")
    asyncio.run(main())
