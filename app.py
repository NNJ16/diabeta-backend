from pickle import TRUE
import pickle
import numpy as np
import pandas as pd
import json
from flask import Flask, request,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Random Forest CLassifier model
prediabetes_model_filename = 'diabetes-prediction-rfc-model.pkl'
prediabetes_model = pickle.load(open(prediabetes_model_filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/diabetes/predict",  methods=['POST'])
def predict_diabetes():
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
    print(prediction)
    print(probability)

    return str(json.dumps({"result": int(prediction[0]), "probability": float(probability[0][1])}))

@app.route("/exercise/recommendation",  methods=['POST'])
def exercise_recommendation():
    df = pd.read_csv("Exercise.csv")
    #remove duplicate exersises
    df.drop_duplicates(subset=['Exercise'], inplace= True)

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
        top5 = list(score.iloc[1:11].index)
        print(top5)
    
        for i in top5:
            exercises.append(df['Exercise'][i])
        return exercises

    df['indexes'] = df['clean_input'].str.find(data_array[0]+data_array[1]+' '+data_array[2]+' '+data_array[3]+' '+data_array[4])
    exercise = df[df['indexes'] >= 0]
    return json.dumps(recommend_exercise(exercise.iloc[0]['Exercise']))

if __name__ == "__main__":
    app.run(debug=TRUE)
