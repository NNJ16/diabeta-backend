from pickle import TRUE
import pickle
import numpy as np
from flask import Flask, request,render_template

# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

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
    prediction = classifier.predict(data)
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=TRUE)
