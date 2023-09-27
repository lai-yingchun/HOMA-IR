import pandas as pd

# 讀取CSV文件
data = pd.read_csv('Data_SEM_before_delete (n=612).csv')
df=pd.DataFrame(data)

# 計算每一列的缺失值數量
missing_values = df.isnull().sum()

# 印出欄位名稱及缺失值
print("欄位名稱\t缺失值數量")
for col in df.columns:
    print(f"{col}\t{missing_values[col]}")
