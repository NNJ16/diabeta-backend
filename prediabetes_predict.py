import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('diabetes_data_upload.csv')

for column in df.columns.drop(['Age','Gender','class']):
     df[column]= df[column].replace({'No':0 , 'Yes': 1})
        
df['class'] = df ['class'].replace({'Positive':1,'Negative':0 })
df['Gender'] = df ['Gender'].replace({'Female':0,'Male':1 })

# Model Building
X = df.drop(['class'], axis='columns')
y = df['class']


X_fs = X[['Age', 'Gender','Polyuria', 'Polydipsia','partial paresis','sudden weight loss','Irritability', 'delayed healing','Alopecia','visual blurring']]

X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size = 0.2,stratify=y, random_state = 1234)

#Standadize Data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Creating Random Forest Model
rf = RandomForestClassifier(criterion='gini',n_estimators=100)
rf.fit(X_train,y_train)

#accuracy score on the training data
t_prediction = rf.predict(X_train)
t_accuracy = accuracy_score(t_prediction, y_train)
print('Accuracy score of the training data: ', t_accuracy)

#accuracy score on the training data
prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print('Accuracy score of the test data: ', accuracy)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(rf, open(filename, 'wb'))