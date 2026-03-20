# summative/linear_regression/predict_consumption.py

import joblib
import numpy as np
import pandas as pd
import os

def load_model_artifacts(model_dir='saved_model'):
    """
    Load the trained model, scaler, and feature names from the specified directory.
    """
    try:
        # Find the model file (assuming there's only one .pkl model file)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        if not model_files:
            raise FileNotFoundError("No model file found in the directory.")
        model_path = os.path.join(model_dir, model_files[0])

        model = joblib.load(model_path)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        features = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        print(f"Model loaded successfully from {model_path}")
        return model, scaler, features
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None

def predict_consumption(input_data, model, scaler, features):
    """
    Predict power consumption for Zone 1 based on input features.

    Args:
        input_data (dict): A dictionary with feature names as keys.
        model: The trained model.
        scaler: The fitted StandardScaler.
        features (list): List of feature names expected by the model.

    Returns:
        float: Predicted power consumption for Zone 1.
    """
    # Create a DataFrame from the input dict, ensuring correct feature order
    input_df = pd.DataFrame([input_data])[features]

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    return prediction[0]

if __name__ == "__main__":
    # Example usage: Predict for a specific set of conditions
    model, scaler, features = load_model_artifacts()

    if model is not None:
        # Example input: A cold winter evening
        sample_input = {
            'Temperature': 5.0,
            'Humidity': 80.0,
            'Wind Speed': 15.0,
            'general diffuse flows': 0.1,
            'Year': 2017,
            'Month': 1,
            'Day': 15,
            'Hour': 20,
            'DayOfWeek': 6  # Sunday
        }

        predicted_consumption = predict_consumption(sample_input, model, scaler, features)
        print(f"\n--- Sample Prediction ---")
        print(f"Input Conditions: {sample_input}")
        print(f"Predicted Power Consumption for Zone 1: {predicted_consumption:.2f}")