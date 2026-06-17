import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CSV_PATH = "data/drowsiness_dataset.csv"
MODEL_SAVE_PATH = "models_ml/drowsiness_model.pkl"

os.makedirs("models_ml", exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[df["status"] != "NO FACE"]

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

# Temporal split — preserves time order so sequential counter features
# (closed_eye_frames, open_mouth_frames) cannot leak between train and test.
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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