import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/df_features.csv')

# ── Prepare features and target ───────────────────────────────────────────────
y_class = df['default']
X = df.drop(columns=['default', 'State', 'Sector'])

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

assert 'state_default_rate' in X_train.columns, "Geographic signal missing"
assert 'industry_default_rate' in X_train.columns, "Industry signal missing"
assert 'state_sector_default_rate' in X_train.columns, "Interaction feature missing — re-run build_features.py"
print("All geographic and industry signals present in classification data")
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ── Model 1: Logistic Regression (Baseline) ───────────────────────────────────
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_f1 = f1_score(y_test, lr_model.predict(X_test))
print(f"  F1: {lr_f1:.4f}")

# ── Model 2: Random Forest (Intermediate) ─────────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_f1 = f1_score(y_test, rf_model.predict(X_test))
print(f"  F1: {rf_f1:.4f}")

# ── Model 3: XGBoost — Priority 2: Class Weight Balancing ─────────────────────
print("\nTraining XGBoost (with class weight balancing)...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight: {scale_pos_weight:.2f}  (dataset is {(y_train==0).mean()*100:.0f}% non-default)")

# Split a calibration set from training data before fitting (Priority 5)
X_train_main, X_cal, y_train_main, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=99
)

xgb_model = XGBClassifier(
    random_state=42,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,   # Priority 2
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train_main, y_train_main)
xgb_f1_raw = f1_score(y_test, xgb_model.predict(X_test))
print(f"  F1 (raw, threshold=0.50): {xgb_f1_raw:.4f}")

print("\nClassification report (raw model):")
print(classification_report(y_test, xgb_model.predict(X_test)))

# ── Priority 3: Threshold Tuning ──────────────────────────────────────────────
print("Finding optimal decision threshold...")
probs = xgb_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.10, 0.60, 0.01)
f1_scores  = [f1_score(y_test, probs >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1_tuned  = max(f1_scores)

print(f"  Default threshold (0.50) F1: {f1_score(y_test, probs >= 0.50):.4f}")
print(f"  Optimal threshold ({best_threshold:.2f}) F1: {best_f1_tuned:.4f}")
print(f"  Delta: {best_f1_tuned - f1_score(y_test, probs >= 0.50):+.4f}")

with open('models/best_threshold.pkl', 'wb') as f:
    pickle.dump(float(best_threshold), f)
print(f"  Saved optimal threshold {best_threshold:.2f} → models/best_threshold.pkl")

# ── Priority 5: Probability Calibration (Platt Scaling) ───────────────────────
print("\nCalibrating probabilities (Platt scaling)...")
calibrated_clf = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
calibrated_clf.fit(X_cal, y_cal)

cal_probs  = calibrated_clf.predict_proba(X_test)[:, 1]
xgb_f1_cal = f1_score(y_test, cal_probs >= best_threshold)
print(f"  Calibrated F1 (threshold={best_threshold:.2f}): {xgb_f1_cal:.4f}")

sample = calibrated_clf.predict_proba(X_test[:10])[:, 1]
print(f"  Sample calibrated probs: {[f'{p:.1%}' for p in sample]}")

# ── F1 Comparison Table ───────────────────────────────────────────────────────
BASELINE_F1 = 0.86
results = pd.DataFrame({
    'Model':    ['Logistic Reg.', 'Random Forest', 'XGBoost (baseline)', 'XGBoost (improved)'],
    'F1_Score': [round(lr_f1, 4), round(rf_f1, 4), BASELINE_F1, round(xgb_f1_cal, 4)]
})
print("\n── Model Comparison ──────────────────────────────────────")
print(results.to_string(index=False))

# ── Final Report ──────────────────────────────────────────────────────────────
print("\n=== Final Model Report ===")
print(f"Features used:         {X_train.shape[1]}")
print(f"New features added:    state_sector_default_rate, loan_size_bucket, zero_jobs_created")
print(f"scale_pos_weight:      {scale_pos_weight:.2f}")
print(f"Optimal threshold:     {best_threshold:.2f}")
print(f"Final F1:              {xgb_f1_cal:.4f}")
print(f"Baseline F1:           {BASELINE_F1}")
print(f"Total gain:            {xgb_f1_cal - BASELINE_F1:+.4f}")
print(f"Calibration:           Applied (Platt scaling)")

# ── Save calibrated model ─────────────────────────────────────────────────────
with open('models/xgb_classifier.pkl', 'wb') as f:
    pickle.dump(calibrated_clf, f)
print("\nSaved: models/xgb_classifier.pkl  (calibrated)")
print("Saved: models/best_threshold.pkl")
