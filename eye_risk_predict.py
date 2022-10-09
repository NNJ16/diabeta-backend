import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('eye_disease.csv')

for column in df.columns.drop(['Age','Gender','class']):
     df[column]= df[column].replace({'No':0 , 'Yes': 1})
        
df['class'] = df ['class'].replace({'Positive':1,'Negative':0 })
df['Gender'] = df ['Gender'].replace({'Female':0,'Male':1 })

X = df.drop(['class'], axis='columns')
y = df['class']

X_fs = X[['Age', 'Gender','Glucoma', 'Surgery','Pain','Vision', 'Diabets']]

X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size = 0.2,stratify=y, random_state = 1234)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

rf = RandomForestClassifier(criterion='gini',n_estimators=100)
rf.fit(X_train,y_train)

t_prediction = rf.predict(X_train)
t_accuracy = accuracy_score(t_prediction, y_train)
print('Accuracy score of the training data: ', t_accuracy)

prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print('Accuracy score of the test data: ', accuracy)

filename = 'eye-prediction-model.pkl'
pickle.dump(rf, open(filename, 'wb'))