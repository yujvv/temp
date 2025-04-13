import os
import duckdb
import pandas as pd
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

# Vanna配置
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        config = config or {}
        config['chroma_persist_directory'] = './.chroma_db'
        os.makedirs(config['chroma_persist_directory'], exist_ok=True)
        
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# API密钥
openai_api_key = "sk-"

# 初始化Vanna
vn = MyVanna(config={
    'api_key': openai_api_key,
    'model': 'gpt-4o'  # 使用你上次输出中显示的模型
})

# 连接DuckDB
conn = duckdb.connect(':memory:')

# 数据目录和文件
data_dir = "data"
csv_files = ["balance_sheet.csv", "cash_flow.csv", "company_overview.csv", 
             "income_statement.csv", "listing_status.csv", 
             "time_series_daily_adjusted.csv", "time_series_daily.csv"]

# 导入CSV文件到DuckDB
for csv_file in csv_files:
    table_name = os.path.splitext(csv_file)[0]
    file_path = os.path.join(data_dir, csv_file)
    
    # 导入数据到DuckDB
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
    
    # 训练查询示例
    vn.train(sql=f"SELECT * FROM {table_name} LIMIT 5")

# 特定训练查询
vn.train(sql="SELECT Symbol, Name, Exchange, MarketCapitalization FROM company_overview WHERE Exchange = 'NASDAQ' ORDER BY MarketCapitalization DESC LIMIT 10")

# 关键步骤：设置Vanna的run_sql方法，正确传递DuckDB连接
def run_sql_function(sql):
    print(f"执行SQL: {sql}")
    return conn.execute(sql).fetchdf()

# 设置run_sql方法 - 这是Vanna执行SQL的关键
vn.run_sql = run_sql_function

# 这一行更改Vanna内部状态，表示已设置run_sql
vn.run_sql_is_set = True

# 示例问题测试
question = "找出市值大于10亿美元且净资产收益率大于5%的十支科技股"
print(f"\n问题: {question}")

# 获取并执行SQL
sql, df, fig = vn.ask(question, print_results=False)

print(f"生成的SQL: {sql}")

# 打印结果
if df is not None and not df.empty:
    print("\n执行SQL结果:")
    print(df)
else:
    print("SQL执行失败或未返回结果")

# 关闭连接
conn.close()