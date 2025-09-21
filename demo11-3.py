from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from transformers.masking_utils import chunked_overlay

model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load() # 这里会去加载网页内容

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

# 指定chain_type 为refine
chain = load_summarize_chain(model, chain_type='refine')
result = chain.invoke(split_docs)
print(result['output_text'])
