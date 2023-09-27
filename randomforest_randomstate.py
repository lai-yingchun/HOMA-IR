import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', None)
data = pd.read_csv('Data_SEM_clear_v1.csv')
df=pd.DataFrame(data)
feature_names = df.keys()
all_feature_names= list(feature_names[2::])
num_feature_names = feature_names[5::]


x_num = df.iloc[:,5::]
standardScaler = StandardScaler()
x_num = pd.DataFrame(standardScaler.fit_transform(x_num),columns=num_feature_names)
x_cat = pd.DataFrame(df.iloc[:,2:5])
df_x = pd.concat([x_cat,x_num], axis=1)
selected_columns = ['group', 'DLCOPre', 'WHR', 'WBC', 'DSBP_A', 'BUN', 'SLmin']
selected_df = df_x[selected_columns]

from sklearn.metrics import precision_score,recall_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve

y =  df['HOMA_IR']
x = selected_df
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=8840)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
test_acc = accuracy_score(y_test, y_pred).round(7)
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm))
print(classification_report(y_test, y_pred))
print('正確率:', test_acc)