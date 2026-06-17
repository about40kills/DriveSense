import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

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

model = joblib.load(MODEL_PATH)

# ── Temporal hold-out evaluation ─────────────────────────────────────────────
# Preserves time order so sequential counter features cannot leak across splits.
split_idx = int(len(df) * 0.8)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("\nTemporal Hold-out Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# ── Cross-validation (k-fold on full dataset) ─────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=False)  # shuffle=False respects time order
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Per-fold scores: {[round(s, 4) for s in cv_scores]}")

# ── Save results ──────────────────────────────────────────────────────────────
with open("results/evaluation_report.txt", "w") as f:
    f.write(f"Temporal Hold-out Accuracy: {accuracy}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"Per-fold scores: {[round(s, 4) for s in cv_scores]}\n")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Drowsiness Detection — Confusion Matrix")
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
