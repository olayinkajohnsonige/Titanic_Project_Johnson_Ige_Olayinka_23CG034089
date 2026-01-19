from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and the scaler/encoder we saved in Part A
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from the form
    pclass = int(request.form['Pclass'])
    sex = request.form['Sex']
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    fare = float(request.form['Fare'])

    # 2. Preprocess the input exactly like we did in training
    sex_encoded = le.transform([sex])[0]
    features = np.array([[pclass, sex_encoded, age, sibsp, fare]])
    features_scaled = scaler.transform(features)

    # 3. Predict
    prediction = model.predict(features_scaled)
    
    # 4. Determine result text
    if prediction[0] == 1:
        result = "Survived"
    else:
        result = "Did Not Survive"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)