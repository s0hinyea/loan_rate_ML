import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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

# Check that encoded features are present before proceeding
assert 'state_default_rate' in X_train.columns, "Geographic signal missing"
assert 'industry_default_rate' in X_train.columns, "Industry signal missing"
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

# ── Model 3: XGBoost (Final Champion) ─────────────────────────────────────────
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    random_state=42,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
xgb_f1 = f1_score(y_test, xgb_model.predict(X_test))
print(f"  F1: {xgb_f1:.4f}")

# ── F1 Comparison Table ───────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model': ['Logistic Reg.', 'Random Forest', 'XGBoost'],
    'F1_Score': [round(lr_f1, 4), round(rf_f1, 4), round(xgb_f1, 4)]
})
print("\n── Model Comparison ──────────────────────────────────────")
print(results.to_string(index=True))

# ── Save XGBoost model ────────────────────────────────────────────────────────
with open('models/xgb_classifier.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("\nSaved: models/xgb_classifier.pkl")
