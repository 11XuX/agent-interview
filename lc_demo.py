"""LangChain demo: JD x 简历 面试助手。Step 5 - batch 批量评估 + RunnableConfig。"""
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
    # DeepSeek V4 默认开 thinking，而 thinking 模式拒绝强制 tool_choice。
    # 解析任务不需要推理，直接关掉：省 token、也让 function_calling 能用。
    extra_body={"thinking": {"type": "disabled"}},
)

JD = """招聘：后端工程师（LLM 应用方向）
- 3 年以上 Python 后端经验
- 熟悉 FastAPI / PostgreSQL / Redis
- 有高并发系统设计与线上排障经验
- 加分：LangChain/LangGraph 等 Agent 框架实践、向量检索、K8s
- 本科及以上，计算机相关专业
"""


class JDSpec(BaseModel):
    """一个招聘岗位的结构化要求。"""

    title: str = Field(description="岗位名称")
    must_have: list[str] = Field(description="硬性要求，逐条拆开，每条只讲一件事")
    nice_to_have: list[str] = Field(description="加分项，逐条拆开")
    min_years: int = Field(description="最低工作年限要求，没写就填 0")


jd_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是资深技术招聘。把 JD 拆成结构化要求，忠于原文，不要脑补原文没有的要求。"),
    ("human", "{jd}"),
])

# method="function_calling"：DeepSeek 不支持 OpenAI 的 json_schema，走 tool calling
# 之后链的出口直接是 JDSpec 实例，不再需要 OutputParser
jd_chain = jd_prompt | llm.with_structured_output(JDSpec, method="function_calling")


RESUMES = {
    p.stem.removeprefix("resume_"): p.read_text(encoding="utf-8")
    for p in sorted((Path(__file__).parent / "data").glob("resume_*.md"))
}


class Profile(BaseModel):
    """一份简历的结构化画像。"""

    name: str = Field(description="候选人姓名")
    years: int = Field(description="总工作年限，按简历中的起止时间推算")
    skills: list[str] = Field(description="明确写出或明确用过的技术，不要推测")
    highlights: list[str] = Field(description="可量化的成绩或有难度的工作，每条一句话")


resume_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是资深技术招聘。把简历拆成结构化画像，只写简历里有的，绝不脑补。"),
    ("human", "{resume}"),
])

resume_chain = resume_prompt | llm.with_structured_output(Profile, method="function_calling")

# RunnableParallel：同一份输入 dict 广播给每个分支，各分支并发跑，输出合并成 dict
parse = RunnableParallel(jd=jd_chain, profile=resume_chain)


class Gap(BaseModel):
    """JD 里某一条要求，在候选人身上的落实情况。"""

    requirement: str = Field(description="JD 里的这条要求，原样抄")
    status: Literal["满足", "存疑", "缺失"] = Field(
        description="满足=简历有直接证据；存疑=有相邻经验但没直接证据；缺失=完全没提"
    )
    evidence: str = Field(description="判断依据，一句话。满足/存疑要指出简历中的哪一条")


class MatchReport(BaseModel):
    """候选人与岗位的匹配评估。"""

    score: int = Field(ge=0, le=100, description="综合匹配分，硬性要求权重远高于加分项")
    verdict: Literal["建议面试", "备选", "不匹配"] = Field(description="结论")
    gaps: list[Gap] = Field(description="逐条评估 JD 的每一条硬性要求，一条都不许漏")
    risks: list[str] = Field(description="值得在面试中当面确认的风险点，没有就给空列表")


match_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是资深技术面试官。只依据简历给出的事实判断，宁可标存疑也不要脑补。"),
    ("human", "岗位要求：\n{jd}\n\n候选人画像：\n{profile}\n\n请逐条评估。"),
])

# LLM 只吃文本，所以上一步的对象要重新序列化回字符串。
# dict 里的裸函数会被自动包成 RunnableLambda，等价于一个 RunnableParallel。
match_chain = (
    {
        "jd": lambda d: d["jd"].model_dump_json(indent=2),
        "profile": lambda d: d["profile"].model_dump_json(indent=2),
    }
    | match_prompt
    | llm.with_structured_output(MatchReport, method="function_calling")
)

# 整条链：并行解析 -> 融合打分。RunnableSequence 里嵌着 RunnableParallel。
chain = parse | match_chain

# .with_retry() 是 Runnable 基类白送的：包一层，整条链的任意一步抛错都会重试。
robust = chain.with_retry(stop_after_attempt=3)

if __name__ == "__main__":
    payloads = [{"jd": JD, "resume": r} for r in RESUMES.values()]

    t = time.perf_counter()
    # max_concurrency 限制同时在飞的请求数，防止打爆厂商的 rate limit
    reports = robust.batch(payloads, config={"max_concurrency": 2})
    print(f"{len(payloads)} 份简历耗时 {time.perf_counter() - t:.1f}s\n")

    for who, r in zip(RESUMES, reports):
        miss = [g.requirement for g in r.gaps if g.status == "缺失"]
        print(f"[{who}] {r.score:3d} 分 / {r.verdict}")
        print(f"      缺失 {len(miss)} 项: {'、'.join(miss) or '无'}")
        print(f"      风险 {len(r.risks)} 条，首条: {r.risks[0] if r.risks else '无'}\n")

    assert len(reports) == len(payloads), "batch 必须一进一出"
    # batch 是并发跑的，但结果按输入顺序返回 —— 这是接口保证，不是巧合
    a, b, c = reports
    assert a.score > c.score, f"强匹配({a.score})该高于弱匹配({c.score})"
    assert a.score > b.score, f"强匹配({a.score})该高于有 gap 的({b.score})"
