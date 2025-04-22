
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Define the feature names (based on your dataset)
feature_names = ['feature1', 'feature2']

# Streamlit UI
st.title("Disease Prediction App")
st.write("Enter the patient details below:")

# Take user input for all features
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:", step=1.0)

# Predict button
if st.button("Predict Disease"):
    input_df = pd.DataFrame([user_input])

    # Ensure input has same feature columns in correct order
    input_df = input_df[feature_names]

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Disease: {prediction}")
