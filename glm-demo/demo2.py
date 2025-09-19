# 使用openai创建
from openai import OpenAI


client = OpenAI(
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

resp = client.chat.completions.create(
    model='glm-4-0520',
    messages=[
        {"role": "user", "content": "北京天气怎么样"},
    ],
    # stream=True
)

#如果是流式输出就需要迭代

print(resp.choices[0].message.content)

