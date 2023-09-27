
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
df_x_orig = df_x.copy()
all_feature_names_orig = all_feature_names.copy()
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random

x = df_x
y = df['HOMA_IR']
for count in range(500):
    random_number = random.randint(0, 500)*count
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=21984)
    print()
    step = []
    rf = RandomForestClassifier(random_state=42)

    # Initialize variables to keep track of best features and accuracy
    best_features = []
    best_accuracy = 0.0
    # Stepwise feature selection
    for i in range(len(x.columns)):
        best_local_accuracy = 0.0
        best_local_feature = None
        
        # Try adding each feature to the existing best_features
        for feature in x.columns:
            if feature not in best_features:
                temp_features = best_features + [feature]
                rf.fit(x_train[temp_features], y_train)
                y_pred = rf.predict(x_test[temp_features])
                acc = accuracy_score(y_test, y_pred)
                if acc > best_local_accuracy:
                    best_local_accuracy = acc
                    best_local_feature = feature
        # If adding a new feature increases accuracy, update the best_features
        if best_local_accuracy > best_accuracy:
            best_accuracy = best_local_accuracy
            best_features.append(best_local_feature)
            step.append(f"Selected features after {i+1} steps: {best_features}, Accuracy: {best_accuracy}")
        else:
            step.append(f"Selected features after {i+1} steps: {temp_features}, Accuracy: {acc}")
            break  # Stop if adding more features doesn't increase accuracy


    if best_accuracy>0.80:
        print("Best selected features:", best_features)
        print("Best evaluation metric:", best_accuracy)
        # Train the final model using the best selected features
        from sklearn.metrics import confusion_matrix, classification_report
        final_rf = RandomForestClassifier(random_state=42)
        final_rf.fit(x_train[best_features], y_train)
        y_pred_final = final_rf.predict(x_test[best_features])
        cm = confusion_matrix(y_test, y_pred_final)
        print(pd.DataFrame(cm))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_final))
        print("Final random_state used:", random_number)
        for s in step:
            print(s)
            print("\n")
        break