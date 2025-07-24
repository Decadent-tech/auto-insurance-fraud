from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)
model = joblib.load(open('model/best_model.pkl', 'rb'))
scaler = joblib.load(open('model/scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        input_features = []
        for i in range(1, 49):  # 1 to 58
            value = request.form.get(f'feature{i}')
            if value == '' or value is None:
                input_features.append(0)  # default to 0 if empty
            else:
                input_features.append(float(value))

        # Convert to DataFrame and scale
        input_array = np.array(input_features).reshape(1, -1)
        scaled_data = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_data)[0]
        result = "Fraud Reported" if prediction == 1 else "No Fraud"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"
@app.route('/fraudtest')
def test_fraud_case():
    fraud_input = np.array([
    7, 2080, 7536453, 73381, 13, 1, 1, 4, 9542, 16238, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0])  # Your 48-value list
    scaled = scaler.transform([fraud_input])
    pred = model.predict(scaled)[0]
    return f"Fraud Prediction Test Case → {'Fraud' if pred == 1 else 'No Fraud'}"
if __name__ == '__main__':
    app.run(debug=True)
