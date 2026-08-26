"""LangChain demo: JD x 简历 面试助手。Step 4 - 融合：把两个对象喂给匹配链。"""
import os
import time
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


RESUME = """张三 / 后端工程师 / 4 年经验
- 某电商公司 2022-2026：Python + Django 做订单系统，日均 200w 订单，主导过一次大促扩容
- 某创业公司 2021-2022：Flask 写内部管理后台，MySQL
- 技能：Python、Django、Flask、MySQL、Docker、RabbitMQ
- 业余用 OpenAI API 写过一个文档问答小工具
- 学历：某双非本科，计算机科学与技术
"""


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

if __name__ == "__main__":
    t = time.perf_counter()
    report = chain.invoke({"jd": JD, "resume": RESUME})
    print(f"耗时 {time.perf_counter() - t:.1f}s\n")

    print(f"{report.score} 分 / {report.verdict}")
    for g in report.gaps:
        print(f"  [{g.status}] {g.requirement} —— {g.evidence}")
    print("风险点:")
    for r in report.risks:
        print(f"  - {r}")

    assert isinstance(report, MatchReport)
    assert 0 <= report.score <= 100
    # 简历用的是 Django/Flask/MySQL，JD 要 FastAPI/PostgreSQL/Redis —— 不可能全满足
    assert any(g.status != "满足" for g in report.gaps), "全标满足了，说明它在脑补"
