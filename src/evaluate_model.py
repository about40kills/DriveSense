import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

CSV_PATH = "data/drowsiness_dataset.csv"
MODEL_PATH = "models_ml/drowsiness_model.pkl"
RESULTS_FOLDER = "results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# Save text results
with open("results/evaluation_report.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}\n\n")
    file.write("Classification Report:\n")
    file.write(report)

# Save confusion matrix image
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Drowsiness Detection Confusion Matrix")
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()