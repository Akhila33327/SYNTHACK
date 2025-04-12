import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Healthcare AI - Disease Prediction & Patient Analysis")

# Upload dataset
@st.cache_data
def load_data():
    df = pd.read_excel("Improved_Chronic_Disease_Dataset.xlsx")
    df = df.dropna()
    for col in ['Gender', 'Symptoms Disease', 'Disease']:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df.drop(columns=[col for col in ['Best_Treatment'] if col in df.columns], inplace=True)
    return df

data = load_data()

# Encode categorical variables
data_encoded = data.copy()
label_encoders = {}
for col in ['Gender', 'Symptoms Disease', 'Disease']:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    label_encoders[col] = le

# Features and target
X = data_encoded.drop(columns=['Disease'])
y = data_encoded['Disease']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Sidebar for patient info
st.sidebar.header("Enter Patient Information")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.sidebar.selectbox("Gender", options=['male', 'female'])
symptom = st.sidebar.selectbox("Symptom", options=data['Symptoms Disease'].unique())

if st.sidebar.button("Predict Disease"):
    if name and symptom:
        # Encode gender and symptom
        try:
            gender_enc = label_encoders['Gender'].transform([gender])[0]
            symptom_enc = label_encoders['Symptoms Disease'].transform([symptom])[0]

            # Predict
            input_df = pd.DataFrame([[age, gender_enc, symptom_enc]], columns=['Age', 'Gender', 'Symptoms Disease'])
            prediction = model.predict(input_df)[0]
            predicted_disease = label_encoders['Disease'].inverse_transform([prediction])[0]

            # Output
            st.subheader("Patient Report")
            st.write(f"*Name:* {name}")
            st.write(f"*Age:* {age}")
            st.write(f"*Gender:* {gender.title()}")
            st.write(f"*Symptom:* {symptom.title()}")
            st.success(f"*Predicted Disease:* {predicted_disease.title()}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please enter both patient name and select a symptom.")