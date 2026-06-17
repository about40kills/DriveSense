"""
Rule-based vs ML classifier comparison on the same temporal test set.

The rule-based system replicates drowsiness_warning.py logic:
  - EAR < 0.20 → increment closed_eye_frames, else reset
  - mouth > 0.07 → increment open_mouth_frames, else reset
  - YAWNING  if open_mouth_frames >= 25
  - DROWSY   if closed_eye_frames >= 10  (and not YAWNING)
  - AWAKE    otherwise

The ML system uses the trained Random Forest on the same test rows.

Both are evaluated against the ground-truth labels in the CSV.
"""
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CSV_PATH      = "data/drowsiness_dataset.csv"
MODEL_PATH    = "models_ml/drowsiness_model.pkl"
RESULTS_DIR   = "results"

# ── Rule-based thresholds (mirrors drowsiness_warning.py) ────────────────────
RB_EAR_THRESH   = 0.20
RB_MOUTH_THRESH = 0.07
RB_EYE_FRAMES   = 10
RB_YAWN_FRAMES  = 25

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load & split data (same temporal split as training) ───────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df["status"] != "NO FACE"].reset_index(drop=True)

split_idx = int(len(df) * 0.8)
test_df   = df.iloc[split_idx:].reset_index(drop=True)

y_true = test_df["status"]

ML_FEATURES = ["left_ear", "right_ear", "avg_ear",
               "mouth_ratio", "closed_eye_frames", "open_mouth_frames"]

# ── ML predictions ────────────────────────────────────────────────────────────
model    = joblib.load(MODEL_PATH)
y_ml     = model.predict(test_df[ML_FEATURES])

# ── Rule-based predictions (recompute counters from raw values) ───────────────
rb_preds = []
rb_eye   = 0
rb_mouth = 0

for _, row in test_df.iterrows():
    rb_eye   = rb_eye   + 1 if row["avg_ear"]     < RB_EAR_THRESH   else 0
    rb_mouth = rb_mouth + 1 if row["mouth_ratio"]  > RB_MOUTH_THRESH else 0

    if rb_mouth >= RB_YAWN_FRAMES:
        rb_preds.append("YAWNING")
    elif rb_eye >= RB_EYE_FRAMES:
        rb_preds.append("DROWSY")
    else:
        rb_preds.append("AWAKE")

y_rb = pd.Series(rb_preds)

# ── Print results ─────────────────────────────────────────────────────────────
labels = ["AWAKE", "DROWSY", "YAWNING"]

print("=" * 60)
print("RULE-BASED SYSTEM  (EAR<0.20, mouth>0.07, frames 10/25)")
print("=" * 60)
rb_acc = accuracy_score(y_true, y_rb)
print(f"Accuracy: {rb_acc:.4f}")
print(classification_report(y_true, y_rb, labels=labels, zero_division=0))

print("=" * 60)
print("ML SYSTEM  (Random Forest, temporal train/test split)")
print("=" * 60)
ml_acc = accuracy_score(y_true, y_ml)
print(f"Accuracy: {ml_acc:.4f}")
print(classification_report(y_true, y_ml, labels=labels, zero_division=0))

# ── Summary table ─────────────────────────────────────────────────────────────
from sklearn.metrics import precision_recall_fscore_support

rb_p, rb_r, rb_f, _ = precision_recall_fscore_support(y_true, y_rb,  labels=labels, zero_division=0)
ml_p, ml_r, ml_f, _ = precision_recall_fscore_support(y_true, y_ml, labels=labels, zero_division=0)

print("=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)
print(f"{'Metric':<28} {'Rule-Based':>12} {'ML Model':>12}")
print("-" * 54)
print(f"{'Overall Accuracy':<28} {rb_acc:>12.4f} {ml_acc:>12.4f}")
for i, cls in enumerate(labels):
    print(f"  {cls} — Precision{'':<10} {rb_p[i]:>12.4f} {ml_p[i]:>12.4f}")
    print(f"  {cls} — Recall{'':<13} {rb_r[i]:>12.4f} {ml_r[i]:>12.4f}")
    print(f"  {cls} — F1{'':<17} {rb_f[i]:>12.4f} {ml_f[i]:>12.4f}")
    print()

# ── Save to file ──────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "comparison_report.txt")
with open(out_path, "w") as f:
    f.write("RULE-BASED vs ML COMPARISON\n")
    f.write(f"Test set size: {len(test_df)} rows (last 20%, temporal split)\n\n")
    f.write(f"{'Metric':<28} {'Rule-Based':>12} {'ML Model':>12}\n")
    f.write("-" * 54 + "\n")
    f.write(f"{'Overall Accuracy':<28} {rb_acc:>12.4f} {ml_acc:>12.4f}\n")
    for i, cls in enumerate(labels):
        f.write(f"  {cls} Precision{'':<12} {rb_p[i]:>12.4f} {ml_p[i]:>12.4f}\n")
        f.write(f"  {cls} Recall{'':<15} {rb_r[i]:>12.4f} {ml_r[i]:>12.4f}\n")
        f.write(f"  {cls} F1{'':<19} {rb_f[i]:>12.4f} {ml_f[i]:>12.4f}\n")

print(f"Saved to {out_path}")
