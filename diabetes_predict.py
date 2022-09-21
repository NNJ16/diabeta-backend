import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report,roc_curve,confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#Removing the rows that contain zero values from Insulin column:
df = df[df.Insulin > 0]
df.isnull().sum()

df['Glucose'].fillna(df['Glucose'].median(), inplace =True)
df['BMI'].fillna(df['BMI'].median(), inplace =True)

df.drop(columns=['DiabetesPedigreeFunction','Pregnancies'], axis=1, inplace = True)

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

reg = LogisticRegression()
reg.fit(X_train,Y_train)   

Y_pred=reg.predict(X_test)
score = accuracy_score(Y_test,Y_pred)
print(score)

# Creating a pickle file for the classifier
filename = 'diabetes-model.pkl'
pickle.dump(reg, open(filename, 'wb'))



