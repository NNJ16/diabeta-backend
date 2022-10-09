import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('heart.csv')

for column in df.columns.drop(['Age','Gender','Diabetes Status','Target']):
     df[column]= df[column].replace({'No':0 , 'Yes': 1})
        
df['Target'] = df ['Target'].replace({'Positive':1,'Negative':0 })
df['Gender'] = df ['Gender'].replace({'Female':0,'Male':1 })
df['Diabetes Status'] = df ['Diabetes Status'].replace({'Yes':1,'No':0 })
df['Blood Pressure'] = df ['Blood Pressure'].replace({'Abnormal':1,'Normal':0 })
df['Cholesterol'] = df ['Cholesterol'].replace({'Abnormal':1,'Normal':0 })

X = df.drop(['Target'], axis='columns')
y = df['Target']

X_fs = X[['Age', 'Gender','Blood Pressure','Cholesterol','Stress Level','Smoking Status','Activitve Level', 'Diabetes Status']]

X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size = 0.2,stratify=y, random_state = 1234)

rf = RandomForestClassifier(criterion='gini',n_estimators=100)
rf.fit(X_train,y_train)

t_prediction = rf.predict(X_train)
t_accuracy = accuracy_score(t_prediction, y_train)
print('Accuracy score of the training data: ', t_accuracy)

prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print('Accuracy score of the test data: ', accuracy)

filename = 'heart-prediction-model.pkl'
pickle.dump(rf, open(filename, 'wb'))