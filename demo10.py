from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import Field, BaseModel


model = ChatOpenAI(
    model='gpt-3.5-turbo',  # 使用支持函数调用的模型
    temperature=0.6,
    api_key='your-openai-api-key',  # 需要 OpenAI API Key
)

class Classification(BaseModel):

    sentiment: str # 情感分类，字符串类型

    aggressive: int = Field(
        description = "攻击性，整数类型，范围0-10",
    )

    language: str = Field(
        description = "语言，字符串类型",
    )

# 创建提示词模板
tagging_prompt = ChatPromptTemplate.from_template(
    '''
    从以下段落提取所需的信息
    只提取Classification类中的字段
    段落: {input}
    '''
)

chain = tagging_prompt | model.with_structured_output(Classification)

resp = chain.invoke({'input': '我今天心情很好，但是有点生气，因为有人骂我傻逼。我说的都是中文，你能听懂吗？'})

print(resp)
