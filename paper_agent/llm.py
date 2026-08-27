"""共享的模型实例。所有节点用同一个。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MAX_CONCURRENCY = 5     # abatch 同时最多几个模型调用

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
    # DeepSeek V4 默认开思考，而思考模式拒绝强制 tool_choice。
    # 结构化输出走 function_calling 必然下发强制 tool_choice，所以要关掉。
    extra_body={"thinking": {"type": "disabled"}},
)
