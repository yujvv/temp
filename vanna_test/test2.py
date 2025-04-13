import os
import duckdb
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
openai_api_key = "sk"

# 初始化Vanna
vn = MyVanna(config={
    'api_key': openai_api_key,
    'model': 'gpt-4o'
})

# 连接DuckDB
conn = duckdb.connect(':memory:')

# 导入CSV文件前先检查字段类型
print("检查公司数据字段类型...")
import pandas as pd
df = pd.read_csv("data/company_overview.csv")
print("MarketCapitalization的原始类型:", df["MarketCapitalization"].dtype)
print("ReturnOnEquityTTM的原始类型:", df["ReturnOnEquityTTM"].dtype)

# 如果是字符串，预处理数据 - 这才是正确的做法
df["MarketCapitalization"] = pd.to_numeric(df["MarketCapitalization"], errors='coerce')
df["ReturnOnEquityTTM"] = pd.to_numeric(df["ReturnOnEquityTTM"], errors='coerce')

print("处理后 MarketCapitalization的类型:", df["MarketCapitalization"].dtype)
print("处理后 ReturnOnEquityTTM的类型:", df["ReturnOnEquityTTM"].dtype)

# 导入预处理的数据
conn.execute("CREATE TABLE company_overview AS SELECT * FROM df")
print("数据已导入并预处理完成")

# 导入其他CSV文件
data_dir = "data"
other_files = ["balance_sheet.csv", "cash_flow.csv", "income_statement.csv", 
               "listing_status.csv", "time_series_daily_adjusted.csv", "time_series_daily.csv"]

for csv_file in other_files:
    if csv_file == "company_overview.csv":
        continue  # 已经处理过
        
    table_name = os.path.splitext(csv_file)[0]
    file_path = os.path.join(data_dir, csv_file)
    
    # 导入数据到DuckDB
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")

# 为所有表获取表结构并训练Vanna
for table in ["company_overview"] + [os.path.splitext(f)[0] for f in other_files]:
    # 获取表结构
    columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
    
    # 构建DDL语句
    column_defs = []
    for col in columns:
        name, type = col[1], col[2]
        column_defs.append(f"{name} {type}")
    
    ddl = f"CREATE TABLE {table} ({', '.join(column_defs)});"
    
    # 训练模型
    print(f"训练表: {table}")
    vn.train(ddl=ddl)
    
    # 训练查询示例
    vn.train(sql=f"SELECT * FROM {table} LIMIT 5")

# 设置执行SQL的函数
def run_sql_function(sql):
    print(f"执行SQL: {sql}")
    return conn.execute(sql).fetchdf()

# 设置run_sql方法
vn.run_sql = run_sql_function
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