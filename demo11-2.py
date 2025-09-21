from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load() # 这里会去加载网页内容

# 第一个stuff
# chain = load_summarize_chain(model, chain_type='stuff')
# result = chain.invoke(docs)
# print(result['output_text'])

prompt_template = '''
    针对下面的内容，写一个简洁的总结：
    {text}
    简洁的总结摘要：
'''
prompt = PromptTemplate.from_template(prompt_template)

chain = {'text':RunnablePassthrough} | model | prompt

stuff_chain = StuffDocumentsChain(llm_chain = LLMChain(llm=model, prompt=prompt), document_variable_name='text')

result = stuff_chain.invoke(docs)
print(result['output_text'])
