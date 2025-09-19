from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'cmdb-upgrade-api'
USERNAME = 'root'

# mysqlclient驱动URL
MYSQL_URI = 'mysql+mysqldb://{}:@{}:{}/{}?charset=utf8mb4'.format(USERNAME, HOSTNAME, PORT, DATABASE)

model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

db = SQLDatabase.from_uri(MYSQL_URI)
chian = create_sql_query_chain(llm=model, db=db)
# chian.get_prompts()[0].pretty_print()

resp = chian.invoke({"question": "请问：一个有多少个主机"})
print(resp)
