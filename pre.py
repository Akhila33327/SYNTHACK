import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("chronic_disease_model.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

# UI
st.title("Chronic Disease Prediction System")
st.write("Enter your details to predict the possible chronic disease.")

# Inputs
age = st.number_input("Enter your Age", min_value=1, max_value=120, step=1)
gender = st.selectbox("Select Gender", ["male", "female"])
symptom = st.text_input("Enter your primary symptom (like fever, cough, headache):").lower().strip()

# Predict button
if st.button("Predict"):
    try:
        # Encode gender and symptom
        gender_encoded = 1 if gender == "male" else 0
        symptom_encoded = symptom_encoder.transform([symptom])[0]

        # Combine features
        features = np.array([[age, gender_encoded, symptom_encoded]])

        # Predict
        prediction = model.predict(features)[0]
        predicted_disease = disease_encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted Disease: {predicted_disease}")
    except Exception as e:
        st.error(f"Prediction failed. Ensure the symptom you typed exists in the dataset.\nError: {e}")