import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CSV_PATH = "data/drowsiness_dataset.csv"
MODEL_SAVE_PATH = "models_ml/drowsiness_model.pkl"

# Create model folder if missing
os.makedirs("models_ml", exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH)

# Remove rows where no face was detected
df = df[df["status"] != "NO FACE"]

# Features used for training
features = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "mouth_ratio",
    "closed_eye_frames",
    "open_mouth_frames"
]

X = df[features]
y = df["status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

print("\n--- Accuracy ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_SAVE_PATH)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")