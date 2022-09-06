import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def predict_diabetes():
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)

    standardized_data = scaler.transform(X)
    X = standardized_data

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, Y_train)

    X_train_prediction = svm_classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score of the training data: ', training_data_accuracy)

    X_test_prediction = svm_classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score of the test data: ', test_data_accuracy)

    input_data = (4,150,92,0,0,37.6,0.191,30)
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    #standardize the input data
    std_data = scaler.transform(input_data_reshape)
    prediction = svm_classifier.predict(std_data)
    return prediction