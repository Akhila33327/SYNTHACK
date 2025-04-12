import pandas as pd
import random

names = [f'Person{i+1}' for i in range(150)]

symptoms_list = [
    "Fever", "Cough", "Cold", "Fatigue", "Shortness of breath",
    "Chest pain", "Nausea", "Vomiting", "Dizziness", "Headache"
]

def generate_symptoms():
    return ", ".join(random.sample(symptoms_list, random.randint(1, 3)))

def generate_bp():
    levels = ["Low (90-100)", "Normal (101-120)", "High (121+)"]
    return random.choice(levels)

def generate_sugar():
    levels = ["Low (<70)", "Normal (70-140)", "High (>140)"]
    return random.choice(levels)

def generate_cholesterol():
    levels = ["Low (<150)", "Normal (150-200)", "High (>200)"]
    return random.choice(levels)

def get_medical_condition(bp, sugar, chol):
    if "High" in bp or "High" in sugar or "High" in chol:
        return random.choice(["Hypertension", "Diabetes", "High Cholesterol"])
    elif "Low" in bp or "Low" in sugar:
        return random.choice(["Hypotension", "Hypoglycemia"])
    else:
        return "Stable"

def get_treatment(condition):
    return {
        "Hypertension": "Antihypertensive Medication",
        "Diabetes": "Insulin Therapy",
        "High Cholesterol": "Statins",
        "Hypotension": "Fluid Therapy",
        "Hypoglycemia": "Glucose Intake",
        "Stable": "No Treatment Required"
    }[condition]

def get_lifestyle(treatment):
    return {
        "Insulin Therapy": "Low sugar diet, regular exercise",
        "Antihypertensive Medication": "Low salt diet, stress management",
        "Statins": "Low fat diet, avoid fried foods",
        "Fluid Therapy": "Increase water intake",
        "Glucose Intake": "Frequent small meals",
        "No Treatment Required": "Maintain current lifestyle"
    }[treatment]

data = []
for name in names:
    bp = generate_bp()
    sugar = generate_sugar()
    chol = generate_cholesterol()
    meds = random.choice(["Yes", "No"])
    allergies = random.choice(["Yes", "No"])
    symptoms = generate_symptoms()
    condition = get_medical_condition(bp, sugar, chol)
    treatment = get_treatment(condition)
    lifestyle = get_lifestyle(treatment)

    data.append({
        "Name": name,
        "BP_Level": bp,
        "Sugar_Level": sugar,
        "Cholesterol_Level": chol,
        "Current_Medications": meds,
        "Allergies": allergies,
        "Symptoms": symptoms,
        "Medical_Condition": condition,
        "Recommended_Treatment": treatment,
        "Expected_Response": "Improved" if condition != "Stable" else "Maintain",
        "Updated_Medication_Suggestion": "Yes" if meds == "No" else "Adjust",
        "Lifestyle_Recommendation": lifestyle
    })

df = pd.DataFrame(data)
df.to_excel("Personalized_Treatment_Dataset_v2.xlsx", index=False)
print("Dataset generated successfully.")