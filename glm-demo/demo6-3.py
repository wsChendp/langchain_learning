"""
新手友好的 SQL 查询链示例
这个程序演示如何将自然语言问题转换为 SQL 查询，执行后生成中文答案
"""

from operator import itemgetter
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# ==================== 配置部分 ====================
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'cmdb-upgrade-api'
USERNAME = 'root'

# 数据库连接字符串
MYSQL_URI = f'mysql+mysqldb://{USERNAME}:@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4'

# 初始化模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

# 连接数据库
db = SQLDatabase.from_uri(MYSQL_URI)

# ==================== 第一步：SQL 生成 ====================
print("🔧 第一步：创建 SQL 生成链...")
create_sql_chain = create_sql_query_chain(llm=model, db=db)

def extract_sql_from_response(response_text: str) -> str:
    """
    从模型的完整响应中提取纯 SQL 语句
    模型返回格式：Question: ... SQLQuery: SELECT ... SQLResult: ...
    我们需要提取 SQLQuery 部分
    """
    if not isinstance(response_text, str):
        return ""
    
    # 清理代码块标记
    cleaned_text = response_text.replace('```sql', '').replace('```', '')
    
    # 只取 SQLQuery 段，避免把 SQLResult 混进去
    sql_part = cleaned_text.split('\nSQLResult:')[0]
    
    # 移除 SQLQuery: 前缀
    pure_sql = sql_part.replace('SQLQuery:', '').strip()
    
    return pure_sql

# 组合：生成完整响应 -> 提取纯 SQL
sql_generator = create_sql_chain | extract_sql_from_response

# ==================== 第二步：SQL 执行 ====================
print("🔧 第二步：创建 SQL 执行工具...")
sql_executor = QuerySQLDatabaseTool(db=db)

# ==================== 第三步：答案生成 ====================
print("🔧 第三步：创建答案生成链...")
answer_template = PromptTemplate.from_template(
    """根据用户问题、SQL查询和查询结果，用中文回答用户的问题。

问题: {question}
SQL查询: {query}
查询结果: {result}

请用自然的中文回答:"""
)

answer_generator = answer_template | model | StrOutputParser()

# ==================== 组合完整流程 ====================
print("🔧 第四步：组合完整流程...")

def build_query_pipeline():
    """
    构建完整的查询管道
    流程：问题 -> 生成SQL -> 执行SQL -> 生成答案
    """
    
    # 第一步：生成 SQL
    # 输入: {"question": "主机中有多少主机架构为'x86'"}
    # 输出: {"question": "...", "query": "SELECT COUNT(*) FROM ..."}
    step1 = RunnablePassthrough.assign(query=sql_generator)
    
    # 第二步：执行 SQL
    # 从第一步结果中提取 'query' 字段，执行 SQL，添加 'result' 字段
    # 输出: {"question": "...", "query": "SELECT ...", "result": [(15,)]}
    step2 = step1.assign(result=itemgetter('query') | sql_executor)
    
    # 第三步：生成最终答案
    # 将包含问题、SQL、结果的数据传递给答案生成器
    complete_pipeline = step2 | answer_generator
    
    return complete_pipeline

# ==================== 执行查询 ====================
if __name__ == "__main__":
    print("🚀 开始执行查询...")
    
    # 构建查询管道
    query_pipeline = build_query_pipeline()
    
    # 用户问题
    user_question = "请问：主机中有多少主机架构为'x86'"
    print(f"📝 用户问题: {user_question}")
    
    # 执行查询
    try:
        result = query_pipeline.invoke({"question": user_question})
        print(f"✅ 查询结果: {result}")
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        
    print("🎉 查询完成！")
