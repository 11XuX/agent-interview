"""PDF Reader：拉全文并按章节切开。

用 Europe PMC 的 fullTextXML 接口，它直接返回带 <sec><title> 标签的
结构化正文，不需要 PDF 解析（PyMuPDF / Docling 那一套全省了）。

arXiv 的 fulltext_url 指向 HTML 摘要页，拿不到结构化正文，暂时只用摘要。
"""

import asyncio
import xml.etree.ElementTree as ET

import httpx

from ..models import Paper, Section
from ..state import State

TIMEOUT = 40
MAX_SECTION_CHARS = 3000     # 单节上限
MAX_PAPER_CHARS = 12000      # 单篇上限

# 只留这几节。Introduction 多是背景铺垫，Supplementary 是附件清单，都不进证据池。
WANTED = ("abstract", "method", "result", "discussion", "conclusion", "evaluation", "benchmark")


def _flat(el) -> str:
    """把一个 XML 元素下的所有文本拍平成一行。"""
    return " ".join("".join(el.itertext()).split())


def _parse(xml: str) -> list[Section]:
    root = ET.fromstring(xml)
    out: list[Section] = []

    if (abs_ := root.find(".//abstract")) is not None:
        out.append(Section(title="Abstract", text=_flat(abs_)[:MAX_SECTION_CHARS]))

    body = root.find(".//body")
    for sec in (body.findall("sec") if body is not None else []):
        title = (sec.findtext("title") or "").strip()
        if not any(w in title.lower() for w in WANTED):
            continue
        out.append(Section(title=title, text=_flat(sec)[:MAX_SECTION_CHARS]))

    # 再压一次总量，超了就砍掉靠后的节
    total = 0
    kept = []
    for s in out:
        if total + len(s.text) > MAX_PAPER_CHARS:
            break
        kept.append(s)
        total += len(s.text)
    return kept


async def _read_one(client: httpx.AsyncClient, p: Paper) -> Paper:
    """拉一篇。失败只记在这一篇上，不往上抛。

    失败隔离放在这里而不是靠 gather(return_exceptions=True)：
    一篇论文拉不到全文是常态（付费墙、格式异常、临时 500），
    不该让其他 9 篇的结果一起作废。
    """
    if "europepmc" not in p.fulltext_url:
        p.read_error = "无结构化全文，只有摘要"
        return p
    try:
        r = await client.get(p.fulltext_url)
        r.raise_for_status()
        p.sections = _parse(r.text)
        if not p.sections:
            p.read_error = "全文里没有目标章节"
    except Exception as e:                      # noqa: BLE001 —— 什么错都不该炸掉整轮
        p.read_error = f"{type(e).__name__}: {e}"[:120]
    return p


async def reader(state: State) -> State:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        papers = await asyncio.gather(*(_read_one(client, p) for p in state["papers"]))
    return {"papers": list(papers)}
