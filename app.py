from pickle import TRUE
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from flask import Flask, request,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

prediabetes_model_filename = 'prediabetes-model.pkl'
prediabetes_model = pickle.load(open(prediabetes_model_filename, 'rb'))

diabetes_model_filename = 'diabetes-model.pkl'
diabetes_model = pickle.load(open(diabetes_model_filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/diabetes/predict",  methods=['POST'])
def predict_diabetes():
    data = request.get_json()

    data_array = [
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["Age"],
    ]
    
    df = pd.read_csv('diabetes.csv')

    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    df = df[df.Insulin > 0]
    df.isnull().sum()

    df['Glucose'].fillna(df['Glucose'].median(), inplace =True)
    df['BMI'].fillna(df['BMI'].median(), inplace =True)

    df.drop(columns=['DiabetesPedigreeFunction','Pregnancies'], axis=1, inplace = True)

    X = df.drop(columns='Outcome', axis=1)

    scaler = StandardScaler()
    scaler.fit(X)
    data = np.array([data_array])
    prediction = diabetes_model.predict(scaler.transform(data))
    probability = diabetes_model.predict_proba(scaler.transform(data))

    return str(json.dumps({"result": int(prediction[0]), "probability": float(probability[0][1])}))

@app.route("/prediabetes/predict",  methods=['POST'])
def predict_prediabetes():
    data = request.get_json()

    data_array = [
        data["Age"],
        data["Gender"],
        data["Polyuria"],
        data["Polydipsia"],
        data["PartialParesis"],
        data["SuddenWeightLoss"],
        data["Irritability"],
        data["DelayedHealing"],
        data["Alopecia"],
        data["VisualBlurring"],
    ]

    data = np.array([data_array])
    prediction = prediabetes_model.predict(data)
    probability = prediabetes_model.predict_proba(data)

    return str(json.dumps({"result": int(prediction[0]), "probability": float(probability[0][1])}))

@app.route("/exercise/recommendation",  methods=['POST'])
def exercise_recommendation():
    df = pd.read_csv("Exercise.csv")

    # convert to lowercase and remove spaces
    def clean(sentence):
        temp = ""
        for word in sentence:
            temp = temp + (word.lower().replace(' ', ''))
        return temp

    df['Lifestyle'] = [clean(x) for x in df['Lifestyle']]
    df['Category'] = [clean(x) for x in df['Category']]
    df['Activity-level'] = [clean(x) for x in df['Activity-level']]
    df['PreDiabetic'] = df ['PreDiabetic'].replace({1:'prediabetic',0:'nonprediabetic' })
    df['Diabetic'] = df ['Diabetic'].replace({1:'diabetic',0:'nondiabetic' })

    # combining all the columns data
    columns = ['Lifestyle', 'Category', 'Activity-level', 'PreDiabetic', 'Diabetic']

    df["life_cat"] = df[['Lifestyle', 'Category']].apply("".join, axis=1)
    df["clean_input"] = df[['life_cat', 'Activity-level', 'PreDiabetic', 'Diabetic']].apply(" ".join, axis=1)
    df["clean_input"]

    df = df[['Exercise', 'clean_input']]
    df.reset_index(inplace=True)
    
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(df['clean_input'])

    cosine_sim = cosine_similarity(features, features)
    
    index = pd.Series(df['Exercise'])
    
    data = request.get_json()

    data_array = [
        data['Lifestyle'],
        data['Category'],
        data['Activity-level'],
        data['PreDiabetic'],
        data['Diabetic'],
    ]

    def recommend_exercise(exercise):
        exercises = []
        idx = index[index == exercise].index[0]
        print(idx)
        score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
        data = list(score.index)
    
    
        for i in data:
            exercises.append(df['Exercise'][i])
        
        # initialize a null list
        unique_list = []
  
        # traverse for all elements
        for x in exercises:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
            
        top10 = unique_list[:10]
    
        return top10

    df['indexes'] = df['clean_input'].str.find(data_array[0]+data_array[1]+' '+data_array[2]+' '+data_array[3]+' '+data_array[4])
    exercise = df[df['indexes'] >= 0]
    if(exercise.empty):
        return ""
    else:
        return json.dumps(recommend_exercise(exercise.iloc[0]['Exercise']))

if __name__ == "__main__":
    app.run(debug=TRUE)
