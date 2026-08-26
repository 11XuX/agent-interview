"""LangChain demo: JD x 简历 面试助手。Step 3 - RunnableParallel 并发解析。"""
import os
import time

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

if __name__ == "__main__":
    payload = {"jd": JD, "resume": RESUME}

    t = time.perf_counter()
    out = parse.invoke(payload)
    parallel_s = time.perf_counter() - t

    t = time.perf_counter()
    jd_chain.invoke(payload)
    resume_chain.invoke(payload)
    serial_s = time.perf_counter() - t

    print(out["jd"].model_dump_json(indent=2))
    print(out["profile"].model_dump_json(indent=2))
    print(f"\n并发 {parallel_s:.1f}s   串行 {serial_s:.1f}s")

    assert set(out) == {"jd", "profile"}, f"RunnableParallel 的 key 应与构造时一致，得到 {set(out)}"
    assert isinstance(out["jd"], JDSpec) and isinstance(out["profile"], Profile)
    assert out["profile"].years == 4, f"年限该是 4，解析成了 {out['profile'].years}"
