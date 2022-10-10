from pickle import TRUE
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from pandas import to_datetime
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from flask import Flask, request,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

prediabetes_model_filename = 'prediabetes-model.pkl'
prediabetes_model = pickle.load(open(prediabetes_model_filename, 'rb'))

diabetes_model_filename = 'diabetes-model.pkl'
diabetes_model = pickle.load(open(diabetes_model_filename, 'rb'))

eyerisk_model_filename = 'eye-prediction-model.pkl'
eyerisk_model = pickle.load(open(eyerisk_model_filename, 'rb'))

kidneyrisk_model_filename = 'kidney-prediction-model.pkl'
kidneyrisk_model = pickle.load(open(kidneyrisk_model_filename, 'rb'))

heartrisk_model_filename = 'heart-prediction-model.pkl'
heartrisk_model = pickle.load(open(heartrisk_model_filename, 'rb'))

def predict_glucose_level(from_date, period, df):
    # define the model
    model = Prophet()

    # fit the model
    model.fit(df)
    
    future = list()
    dates = pd.date_range(from_date, periods=period, freq='D')

    for day in dates:
        date = str(day.date())
        future.append([date])
	
    future = DataFrame(future)
    future.columns = ['ds']
    future['ds'] = to_datetime(future['ds'])

    #use the model to make a forecast
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    print(forecast)
    return forecast.to_json(orient='index')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/glucose/predict",  methods=['POST'])
def predict_glucose():
    data = request.get_json()
    df = pd.json_normalize(data["glucose"])
    date = data["fromDate"]
    df.columns = ['ds', 'y']
    df['ds']= to_datetime(df['ds'])
    forecast = predict_glucose_level(date, 3, df)
    return forecast

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

@app.route("/eyerisk/predict",  methods=['POST'])
def predict_eyerisk():
    data = request.get_json()

    data_array = [
        data["Age"],
        data["Gender"],
        data["Glucoma"],
        data["Surgery"],
        data["Pain"],
        data["Vision"],
        data["Diabetes"]
    ]

    data = np.array([data_array])
    prediction = eyerisk_model.predict(data)
    probability = eyerisk_model.predict_proba(data)

    return str(json.dumps({"result": int(prediction[0]), "probability": float(probability[0][1])}))

@app.route("/heartrisk/predict",  methods=['POST'])
def predict_heartrisk():
    data = request.get_json()

    data_array = [
        data["Age"],
        data["Gender"],
        data["Blood Pressure"],
        data["Cholesterol"],
        data["Stress Level"],
        data["Smoking Status"],
        data["Activitve Level"],
        data["Diabetes Status"]
    ]

    data = np.array([data_array])
    prediction = heartrisk_model.predict(data)
    probability = heartrisk_model.predict_proba(data)

    return str(json.dumps({"result": int(prediction[0]), "probability": float(probability[0][1])}))

@app.route("/kidneyrisk/predict",  methods=['POST'])
def predict_kidneyrisk():
    data = request.get_json()

    data_array = [
        data["Age"],
        data["Gender"],
        data["Take Medicine"],
        data["Kidney Problem"],
        data["Urine Low"],
        data["Diabetes"]
    ]

    data = np.array([data_array])
    prediction = kidneyrisk_model.predict(data)
    probability = kidneyrisk_model.predict_proba(data)

    return str(json.dumps({"result": int(prediction[0]), "probability": float(probability[0][1])}))


if __name__ == "__main__":
    app.run(debug=TRUE)
