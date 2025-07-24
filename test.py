import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')

fraud_found = False
attempt = 0

while not fraud_found:
    attempt += 1

    # Generate synthetic input (shape: 1 x 48)
    # You can tweak the ranges to increase fraud likelihood
    input_array = np.array([[ 
        np.random.randint(1, 60),           # months_as_customer
        np.random.randint(500, 2500),       # policy_deductable
        np.random.randint(0, 10000000),     # umbrella_limit
        np.random.randint(1000, 100000),    # total_claim
        np.random.randint(0, 24),           # incident_hour_of_the_day
        np.random.randint(0, 5),            # number_of_vehicles_involved
        np.random.randint(0, 3),            # bodily_injuries
        np.random.randint(0, 5),            # witnesses
        np.random.randint(0, 50000),        # capital-gains
        np.random.randint(0, 50000),        # capital-loss

        *np.random.choice([0, 1], size=38),  # 38 one-hot encoded categorical features
    ]])

    # Scale input
    scaled = scaler.transform(input_array)

    # Predict
    pred = model.predict(scaled)[0]

    if pred == 1:
        print(f"\n🎯 Fraud case found on attempt #{attempt}!\n")
        print("Input array:")
        print(input_array)
        print("\nPrediction: Fraud")
        fraud_found = True
    else:
        print(f"Attempt #{attempt} -> No Fraud")

