import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure the artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

print("Loading Student Lifestyle and Stress Dataset...")
df = pd.read_csv("dataset.csv")

# Separate features and target variable
X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]

num_features = ['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure', 'Family_Support', 'Month']
cat_features = ['Student_Type']

print("Building Preprocessing Pipelines...")
# ADDED: StandardScaler is required for Logistic Regression and SVC
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("Splitting data into train and test sets...\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- The Algorithmic Tournament ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

best_model_name = ""
best_model_score = 0.0
best_pipeline = None

print("🏆 Starting Model Training Tournament...\n")

for model_name, classifier in models.items():
    # Build a temporary pipeline for each model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🔹 {model_name} Accuracy: {acc * 100:.2f}%")
    
    # Check if this is the new champion
    if acc > best_model_score:
        best_model_score = acc
        best_model_name = model_name
        best_pipeline = pipeline

print("-" * 50)
print(f"🥇 WINNER: {best_model_name} with {best_model_score * 100:.2f}% Accuracy")
print("-" * 50)

print("\nDetailed Performance Report for the Champion:")
y_pred_best = best_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_best))

print("\nSerializing the champion model architecture to /artifacts...")
with open("artifacts/stress_pipeline_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("Training complete! The best model is ready for deployment.")