import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/df_features.csv')

# ── Prepare features and target ───────────────────────────────────────────────
y_reg = df['sba_coverage_ratio']

# Drop target components (prevent mathematical leakage), classification target, and strings
X = df.drop(columns=['sba_coverage_ratio', 'SBA_Appv', 'GrAppv', 'default', 'State', 'Sector'])

assert 'state_default_rate' in X.columns, "Geographic signal missing"
assert 'industry_default_rate' in X.columns, "Industry signal missing"
print("All geographic and industry signals present")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ── Train XGBoost Regressor ───────────────────────────────────────────────────
print("\nTraining XGBoost Regressor...")
xgb_reg = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_reg.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = xgb_reg.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"XGBoost Regressor R2:   {r2:.3f}")
print(f"XGBoost Regressor RMSE: {rmse:.3f}")

# ── Save model ────────────────────────────────────────────────────────────────
with open('models/xgb_regressor.pkl', 'wb') as f:
    pickle.dump(xgb_reg, f)
print("\nSaved: models/xgb_regressor.pkl")
