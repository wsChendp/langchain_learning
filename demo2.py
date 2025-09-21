import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_openai import ChatOpenAI
from langserve import add_routes


os.environ["LANGCHAIN_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,  # 修正为数字类型
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个有记忆的AI助手。你必须仔细阅读对话历史，记住用户提供的所有信息。用{language}回答问题。'),
    MessagesPlaceholder(variable_name='my_msg')
])

# 得到链
chain = prompt_template | model

# 保存聊天的历史记录
store = {}  # 所有用户的聊天记录都保存到store。key: sessionId,value: 历史聊天记录对象


# 此函数预期将接收一个session_id并返回一个消息历史记录对象。
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'  # 每次聊天时候发送msg的key
)

config = RunnableConfig(configurable={'session_id': 'zs1234'})  # 给当前会话定义一个sessionId


# 第一轮
resp1 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='你好啊！ 我是LaoXiao')],
        'language': '中文'
    },
    config=config
)

print(resp1.content)

# 第二轮
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='请问：我和你上轮对话时候，我的名字是什么？')],
        'language': '中文'
    },
    config=config
)

print(resp2.content)

# 第3轮： 返回的数据是流式的
config = RunnableConfig(configurable={'session_id': '123'})  # 给当前会话定义一个sessionId
for resp in do_message.stream({'my_msg': [HumanMessage(content='请给我讲一个的笑话？')], 'language': '中文'},
                              config=config):
    # 每一次resp都是一个token
    print(resp.content, end='-')
