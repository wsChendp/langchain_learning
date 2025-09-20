import os
import shutil

import lancedb
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 设置 API Key
os.environ['BAICHUAN_API_KEY'] = 'sk-c3cb1ff363cd14e257a673bc3d9e0d16'

print("📚 开始构建 RAG 系统...")

# 加载文档
print("📖 加载文档...")
loader = TextLoader(file_path="test.txt", encoding='utf-8')
documents = loader.load()
print(f"✅ 加载了 {len(documents)} 个文档")

# 文本分割
print("✂️ 分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # 增加块大小
    chunk_overlap=50,  # 增加重叠
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
print(f"✅ 分割成 {len(docs)} 个文档块")

# 使用百川嵌入模型（更稳定）
print("🧠 初始化嵌入模型...")
embeddings = BaichuanTextEmbeddings()

# 清理旧的数据库
db_path = os.path.join(os.getcwd(), "lanceDB")
if os.path.exists(db_path):
    print("🗑️ 清理旧的数据库...")
    shutil.rmtree(db_path)

# 连接向量数据库
print("💾 创建向量数据库...")
db = lancedb.connect(db_path)

# 创建向量存储
print("🔄 生成向量嵌入并存储...")
vectorStore = LanceDB.from_documents(
    docs,
    embeddings,
    connection=db,
    table_name="my_vectors"
)
print("✅ 向量数据库创建完成！")

# ==================== RAG 查询部分 ====================
print("\n🔍 开始 RAG 查询测试...")

# 测试查询
query = ('美国首次申请失业救济人数增加了多少？上一次新高是什么时候')
print(f"📝 查询问题: {query}")

# 测试向量相似性搜索
print("🔎 执行相似性搜索...")
docs_and_scores = vectorStore.similarity_search_with_score(query, k=3)
print(f"✅ 找到 {len(docs_and_scores)} 个相关文档")

# 显示检索到的文档
for i, (doc, score) in enumerate(docs_and_scores):
    print(f"\n📄 文档 {i+1} (相似度: {score:.4f}):")
    print(f"内容: {doc.page_content[:100]}...")

# 创建检索器
print("\n🔧 创建 RAG 链...")
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

# 定义提示词模板
template = """你是一个有帮助的助理。根据以下提供的文档来回答问题。如果你不确定答案，请说你不知道，而不是编造答案。

文档内容:
{context}

问题: {question}

请基于文档内容回答:"""

prompt = ChatPromptTemplate.from_template(template)

# 创建模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,  # 修正为数字类型
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

output_parser = StrOutputParser()

# 构建 RAG 链
# 并行处理：同时获取检索结果和用户问题
rag_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | output_parser
)

# 执行查询
print("\n🚀 执行 RAG 查询...")
try:
    response = rag_chain.invoke(query)
    print(f"\n✅ RAG 回答:\n{response}")
except Exception as e:
    print(f"❌ 查询失败: {e}")

print("\n🎉 RAG 系统测试完成！")
