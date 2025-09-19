import os

from zhipuai import ZhipuAI

# api_key = os.getenv('API_KEY')

client = ZhipuAI(api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q')

resp = client.chat.completions.create(
    model='glm-4-0520',
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"},
    ],
    # stream=True
)

print(resp.choices[0].message.content)

