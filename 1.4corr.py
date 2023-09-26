import pandas as pd
data = pd.read_csv('Data_SEM_1.4.csv')
df=pd.DataFrame(data)

# print(df.info())

print(df.corr()['HOMA_IR1.4'].sort_values())
