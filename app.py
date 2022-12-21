from flask import Flask, request
import joblib
import numpy as np
import json

# URL
# app https://vid-heart-disease-prediction.herokuapp.com/

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Vidhi Patel"

@app.route('/', methods=['POST'])
def heart_disease_pred():
    model = joblib.load("./HEART_DISEASE_MODEL.pkl")

    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    result = model.predict(input)[0]
    
    return json.dumps({"heart_disease":str(result)})

if __name__ == "__main__":
    app.run(debug=True)