"""LangChain demo: JD x 简历 面试助手。Step 2 - 结构化输出。"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
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

if __name__ == "__main__":
    spec = jd_chain.invoke({"jd": JD})
    print(spec.model_dump_json(indent=2))

    assert isinstance(spec, JDSpec), f"没拿到 JDSpec，拿到的是 {type(spec)}"
    assert spec.must_have, "硬性要求解析为空"
    assert spec.min_years == 3, f"年限该是 3，解析成了 {spec.min_years}"
