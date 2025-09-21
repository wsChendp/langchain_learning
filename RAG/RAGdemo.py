import os

import lancedb
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import BaichuanTextEmbeddings, ModelScopeEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['BAICHUAN_API_KEY'] = 'sk-c3cb1ff363cd14e257a673bc3d9e0d16'
loader = TextLoader(file_path="test.txt", encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        "。", "！", "？", ".", "!", "?",
        ",", "，", ";", "；",
        "、", " ",
        ""
    ]
)
docs = text_splitter.split_documents(documents)
print('=====')
embeddings = ModelScopeEmbeddings(model_id="iic/nlp_gte_sentence-embedding_chinese-base")
# 连接向量数据库
db = lancedb.connect(os.path.join(os.getcwd(),"lanceDB"))

vectorStore = LanceDB.from_documents(docs, embeddings, connection=db, table_name="my_vectors")

query = '美国首次申请失业救济人数增加了多少？'

# 测试向量数据库
docs_and_scores = vectorStore.similarity_search_with_score(query)

retriever = vectorStore.as_retriever()
template = """你是一个有帮助的助理。根据以下提供的文档来回答问题。如果你不确定答案，请说你不知道，而不是编造答案。
    
    {context}
    
    问题: {question}
    
    回答: 
    """

prompt = ChatPromptTemplate.from_template(template)

# 创建模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature='0.6',
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

output_parser = StrOutputParser()

# 把检索器和用户输入的问题，结合得到检索结果
start_retriever = RunnableParallel({'context': retriever, 'question': RunnablePassthrough()})

# 创建长链
long_chain = start_retriever | prompt | model | output_parser

resp = long_chain.invoke(query)

print(resp)
