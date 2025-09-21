import os

from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI


os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["TAVILY_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature='0.6',
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
# result = chain(  # 给于一些关键词， 随机生成一句话
#     {
#         "fields": ['蓝色', '黄色'],
#         "preferences": {}
#     }
# )

result = chain(  # 给于一些关键词， 随机生成一句话
    {
        "fields": {"颜色": ['蓝色', '黄色']},
        "preferences": {"style": "让它像诗歌一样。"}
    }
)
print(result)

