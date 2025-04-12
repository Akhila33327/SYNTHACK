import streamlit as st
import joblib
import numpy as np

# Load models and encoders
model_condition = joblib.load("condition_model.pkl")
model_treatment = joblib.load("treatment_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoders = joblib.load("target_encoders.pkl")
mlb = joblib.load("symptom_encoder.pkl")

st.title("AI Health Advisor")
st.subheader("Enter Patient Details")

bp_level = st.selectbox("BP Level", label_encoders['BP_Level'].classes_)
sugar_level = st.selectbox("Sugar Level", label_encoders['Sugar_Level'].classes_)
cholesterol_level = st.selectbox("Cholesterol Level", label_encoders['Cholesterol_Level'].classes_)
medications = st.selectbox("Current Medications", label_encoders['Current_Medications'].classes_)
allergies = st.selectbox("Known Allergies", label_encoders['Allergies'].classes_)
selected_symptoms = st.multiselect("Select Symptoms", mlb.classes_)

def encode_input(val, col):
    return label_encoders[col].transform([val])[0]

categorical_input = [
    encode_input(bp_level, 'BP_Level'),
    encode_input(sugar_level, 'Sugar_Level'),
    encode_input(cholesterol_level, 'Cholesterol_Level'),
    encode_input(medications, 'Current_Medications'),
    encode_input(allergies, 'Allergies')
]

symptom_input = mlb.transform([selected_symptoms])[0]
input_data = np.concatenate([categorical_input, symptom_input]).reshape(1, -1)

if st.button("Predict"):
    pred_condition = model_condition.predict(input_data)[0]
    pred_treatment = model_treatment.predict(input_data)[0]

    condition = target_encoders['Medical_Condition'].inverse_transform([pred_condition])[0]
    treatment = target_encoders['Recommended_Treatment'].inverse_transform([pred_treatment])[0]

    st.subheader("Prediction Results")
    st.write(f"*Medical Condition:* {condition}")
    st.write(f"*Recommended Treatment:* {treatment}")

    st.subheader("Lifestyle Tips")
    if "Diabetes" in condition:
        st.write("• Low-sugar diet, whole grains, regular walks.")
    elif "Hypertension" in condition:
        st.write("• Low-sodium meals, daily exercise, stress control.")
    elif "Cholesterol" in condition:
        st.write("• Avoid fried food, eat fiber-rich diet.")
    else:
        st.write("• Maintain a balanced diet and active lifestyle.")