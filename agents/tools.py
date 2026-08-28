"""三个版本共用的工具。对照实验要控制变量，工具必须完全一样。"""

import asyncio
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ATTEMPTS = 3


async def _get(url: str, params: dict | None = None, timeout: int = 30) -> httpx.Response | str:
    """带重试的 GET。失败返回**字符串**而不是抛异常。

    工具抛异常会炸掉整个 agent loop，模型连"这次没查到"都不知道。
    返回错误文本的话模型能看到、能换个查法重试 —— 这是工具设计的基本原则：
    错误信息要能指导下一步动作。
    """
    last = ""
    for i in range(ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                return await c.get(url, params=params)
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:100]
            if i < ATTEMPTS - 1:
                await asyncio.sleep(2 ** i)
    return f"请求失败（重试 {ATTEMPTS} 次）：{last}。可以换个检索式再试。"


@tool
async def search_papers(query: str) -> str:
    """按检索式查 Europe PMC 开放获取文献。

    检索式用英文，核心概念之间用 AND，同义词用 OR 括起来。
    例：(scRNA-seq OR "single-cell RNA-seq") AND "batch effect" AND (benchmark OR comparison)

    返回每篇的 PMCID、年份、标题、摘要前 300 字。
    """
    r = await _get(f"{EPMC}/search", {
        "query": f"({query}) AND OPEN_ACCESS:Y", "format": "json",
        "pageSize": 5, "resultType": "core"}, timeout=25)
    if isinstance(r, str):
        return r
    if r.status_code != 200:
        return f"检索失败 HTTP {r.status_code}，检索式可能有语法问题，换一个试试"
    out = [
        f"{it.get('pmcid', '?')} ({it.get('pubYear')}) {it.get('title', '')[:80]}\n"
        f"  {(it.get('abstractText') or '无摘要')[:300]}"
        for it in r.json().get("resultList", {}).get("result", [])
    ]
    return "\n".join(out) or "没查到"


@tool
async def read_section(pmcid: str, section: str) -> str:
    """读某篇论文的某一节全文。section 传 Methods / Results / Discussion / Conclusions 等。

    只在摘要不足以判断时才用 —— 一次调用会返回两三千字，很占上下文。
    """
    r = await _get(f"{EPMC}/{pmcid}/fullTextXML", timeout=45)
    if isinstance(r, str):
        return r
    if r.status_code != 200:
        return f"{pmcid} 拿不到全文（HTTP {r.status_code}），换一篇"
    try:
        body = ET.fromstring(r.text).find(".//body")
    except ET.ParseError:
        return f"{pmcid} 全文格式异常，解析不了，换一篇"
    for sec in (body.findall("sec") if body is not None else []):
        title = (sec.findtext("title") or "").strip()
        if section.lower() in title.lower():
            return f"[{pmcid} · {title}]\n" + " ".join("".join(sec.itertext()).split())[:2500]
    have = [(s.findtext("title") or "?").strip() for s in (body.findall("sec") if body is not None else [])]
    return f"{pmcid} 没有叫 {section} 的章节。有这些：{have}"


TOOLS = [search_papers, read_section]
