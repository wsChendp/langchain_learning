"""
超简单的 SQL 查询链示例 - 分步执行版本
适合新手理解每个步骤的作用
"""

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ==================== 配置 ====================
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'cmdb-upgrade-api'
USERNAME = 'root'
MYSQL_URI = f'mysql+mysqldb://{USERNAME}:@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4'

# 初始化模型和数据库
model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)
db = SQLDatabase.from_uri(MYSQL_URI)

# ==================== 工具函数 ====================
def extract_sql(text: str) -> str:
    """从模型响应中提取纯 SQL 语句"""
    if not text:
        return ""
    
    # 清理代码块标记
    text = text.replace('```sql', '').replace('```', '')
    
    # 只取 SQLQuery 部分
    if 'SQLQuery:' in text:
        sql_part = text.split('SQLQuery:')[1]
        if 'SQLResult:' in sql_part:
            sql_part = sql_part.split('SQLResult:')[0]
        return sql_part.strip()
    
    return text.strip()

# ==================== 分步执行 ====================
def step_by_step_query(question: str):
    """
    分步执行查询，让新手理解每个步骤
    """
    print(f"🔍 用户问题: {question}")
    print("=" * 50)
    
    # 第一步：生成 SQL
    print("📝 第一步：生成 SQL 查询...")
    create_sql_chain = create_sql_query_chain(llm=model, db=db)
    
    # 调用 SQL 生成链
    sql_response = create_sql_chain.invoke({"question": question})
    print(f"模型原始响应:\n{sql_response}")
    print("-" * 30)
    
    # 提取纯 SQL
    pure_sql = extract_sql(sql_response)
    print(f"提取的 SQL: {pure_sql}")
    print("-" * 30)
    
    # 第二步：执行 SQL
    print("⚡ 第二步：执行 SQL 查询...")
    sql_executor = QuerySQLDatabaseTool(db=db)
    
    try:
        query_result = sql_executor.invoke(pure_sql)
        print(f"查询结果: {query_result}")
    except Exception as e:
        print(f"SQL 执行失败: {e}")
        return
    print("-" * 30)
    
    # 第三步：生成答案
    print("💬 第三步：生成中文答案...")
    answer_template = PromptTemplate.from_template(
        """根据用户问题、SQL查询和查询结果，用中文回答用户的问题。

问题: {question}
SQL查询: {query}
查询结果: {result}

请用自然的中文回答:"""
    )
    
    answer_chain = answer_template | model | StrOutputParser()
    
    # 准备答案生成的输入
    answer_input = {
        "question": question,
        "query": pure_sql,
        "result": str(query_result)
    }
    
    final_answer = answer_chain.invoke(answer_input)
    print(f"最终答案: {final_answer}")
    print("=" * 50)

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("🚀 开始分步执行 SQL 查询...")
    
    # 测试问题
    test_question = "请问：主机中有多少主机架构为'x86'"
    
    try:
        step_by_step_query(test_question)
        print("✅ 查询完成！")
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    
    print("\n" + "="*60)
    print("💡 理解要点:")
    print("1. 第一步：模型根据问题生成 SQL")
    print("2. 第二步：执行 SQL 获取数据")
    print("3. 第三步：模型根据结果生成中文答案")
    print("4. 每个步骤都可以独立调试和测试")
    print("="*60)
