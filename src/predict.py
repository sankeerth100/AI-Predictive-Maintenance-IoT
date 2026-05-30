import joblib
import pandas as pd

def predict_failure(input_data, scaler, feature_columns):
    model = joblib.load("models/predictive_model.pkl")

    input_df = pd.DataFrame([input_data], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        alert = "ALERT: Machine failure risk detected!"
    else:
        alert = "Machine is operating normally."

    return prediction, probability, alert