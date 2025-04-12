import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_excel("Personalized_Treatment_Dataset_v2.xlsx")

# Encode categorical features
categorical_cols = ['BP_Level', 'Sugar_Level', 'Cholesterol_Level', 'Current_Medications', 'Allergies']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Process symptoms
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip() for s in x.split(',')])
mlb = MultiLabelBinarizer()
symptom_features = mlb.fit_transform(df['Symptoms'])
symptom_df = pd.DataFrame(symptom_features, columns=mlb.classes_)
df = pd.concat([df, symptom_df], axis=1)

# Encode target labels
target_encoders = {}
target_cols = ['Medical_Condition', 'Recommended_Treatment']
for col in target_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    target_encoders[col] = le

# Train/test split
features = categorical_cols + list(mlb.classes_)
X = df[features]
y_condition = df['Medical_Condition']
y_treatment = df['Recommended_Treatment']

X_train, X_test, y_cond_train, y_cond_test, y_trt_train, y_trt_test = train_test_split(
    X, y_condition, y_treatment, test_size=0.2, random_state=42
)

# Train models
model_condition = RandomForestClassifier(random_state=42)
model_treatment = RandomForestClassifier(random_state=42)

model_condition.fit(X_train, y_cond_train)
model_treatment.fit(X_train, y_trt_train)

# Evaluate
print("Condition accuracy:", accuracy_score(y_cond_test, model_condition.predict(X_test)))
print("Treatment accuracy:", accuracy_score(y_trt_test, model_treatment.predict(X_test)))

# Save models
joblib.dump(model_condition, "condition_model.pkl")
joblib.dump(model_treatment, "treatment_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoders, "target_encoders.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")

print("Models trained and saved.")