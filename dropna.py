import pandas as pd

# 讀取a.csv
data = pd.read_csv('Data_SEM_before_delete (n=612).csv')
df=pd.DataFrame(data)
# 刪除包含空值的行
df_cleaned = df .dropna()

# 保存處理後的數據到b.csv
df_cleaned.to_csv('Data_SEM_clear_v1.csv', index=False)

print(".csv 已生成.")
