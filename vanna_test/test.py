import os
import duckdb
import pandas as pd
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

# Vanna配置
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        
    # 重写ask方法来确保返回SQL
    def ask(self, question):
        sql = super().ask(question)
        if sql is None:
            # 如果返回None，手动从LLM响应中提取SQL
            print("从LLM响应中提取SQL...")
            return self.generate_sql(question)
        return sql

# 替换为你的API密钥
openai_api_key = "sk"

# 初始化Vanna
vn = MyVanna(config={
    'api_key': openai_api_key,
    'model': 'gpt-4'
})

# 连接数据库
def connect_database():
    # 创建内存中的DuckDB连接
    conn = duckdb.connect(':memory:')
    
    # 数据目录和文件
    data_dir = "data"
    csv_files = ["balance_sheet.csv", "cash_flow.csv", "company_overview.csv", 
                "income_statement.csv", "listing_status.csv", 
                "time_series_daily_adjusted.csv", "time_series_daily.csv"]
    
    # 导入所有CSV文件到DuckDB
    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0]
        file_path = os.path.join(data_dir, csv_file)
        
        # 导入数据
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
        
        # 获取表结构
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        
        # 构建DDL语句
        column_defs = []
        for col in columns:
            name, type = col[1], col[2]
            column_defs.append(f"{name} {type}")
        
        ddl = f"CREATE TABLE {table_name} ({', '.join(column_defs)});"
        
        # 训练模型
        print(f"训练表: {table_name}")
        vn.train(ddl=ddl)
        
        # 训练一个简单的查询示例
        vn.train(sql=f"SELECT * FROM {table_name} LIMIT 5")
        
        # 如果是company_overview表，添加特定查询
        if table_name == "company_overview":
            vn.train(sql="SELECT Symbol, Name, Exchange, MarketCapitalization FROM company_overview WHERE Exchange = 'NASDAQ' ORDER BY MarketCapitalization DESC LIMIT 10")
    
    return conn

# 连接数据库并训练模型
conn = connect_database()

# 设置vn可以访问数据库连接
vn.conn = conn

# 示例问题测试
question = "找出纳斯达克的十支市值最高的股票"
print(f"\n问题: {question}")

# 获取SQL
sql = vn.ask(question)
print(f"生成的SQL: {sql}")

# 如果SQL生成成功，尝试执行
if sql:
    try:
        print("\n执行SQL结果:")
        result = conn.execute(sql).fetchdf()
        print(result)
    except Exception as e:
        print(f"执行SQL时出错: {e}")
else:
    print("未能生成有效的SQL查询")

# 关闭连接
conn.close()