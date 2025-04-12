import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_excel("Personalized_Treatment_Dataset_v2.xlsx")

# Encode categorical features
label_encoders = {}
categorical_cols = ['BP_Level', 'Sugar_Level', 'Cholesterol_Level',
                    'Current_Medications', 'Allergies']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target columns
target_cols = ['Medical_Condition', 'Recommended_Treatment']
target_encoders = {}

for col in target_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    target_encoders[col] = le

# Features and labels
features = ['BP_Level', 'Sugar_Level', 'Cholesterol_Level',
            'Current_Medications', 'Allergies']
X = df[features]
y_condition = df['Medical_Condition']
y_treatment = df['Recommended_Treatment']

# Train/test split
X_train, X_test, y_cond_train, y_cond_test, y_trt_train, y_trt_test = train_test_split(
    X, y_condition, y_treatment, test_size=0.2, random_state=42
)

# Train models
model_condition = RandomForestClassifier(random_state=42)
model_treatment = RandomForestClassifier(random_state=42)

model_condition.fit(X_train, y_cond_train)
model_treatment.fit(X_train, y_trt_train)

# Save models
joblib.dump(model_condition, "condition_model.pkl")
joblib.dump(model_treatment, "treatment_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoders, "target_encoders.pkl")

print("Models trained and saved successfully.")