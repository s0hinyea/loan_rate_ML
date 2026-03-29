import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import shap

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs('visuals', exist_ok=True)

# ── Load data and model ───────────────────────────────────────────────────────
df = pd.read_csv('data/df_features.csv')

X = df.drop(columns=['default', 'State', 'Sector'])

with open('models/xgb_classifier.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# ── Visual 1: SHAP Summary Plot ───────────────────────────────────────────────
print("Generating SHAP summary plot...")
X_sample = X.sample(n=5000, random_state=42)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("XGBoost Feature Importance (SHAP)", fontsize=16)
plt.tight_layout()
plt.savefig('visuals/shap_summary.png', dpi=300)
plt.close()
print("Saved: visuals/shap_summary.png")

# ── Visual 2: Time Series — Default Rate Over Time ────────────────────────────
print("Generating time series chart...")
yearly_defaults = df.groupby('ApprovalFY')['default'].mean()

fig, ax = plt.subplots(figsize=(12, 5))
yearly_defaults.plot(ax=ax, color='steelblue', linewidth=2.5)

ax.axvspan(2008, 2010, alpha=0.2, color='red', label='2008 Financial Crisis')

ax.set_title('Small Business Default Rate Over Time', fontsize=16, fontweight='bold')
ax.set_ylabel('Historical Default Rate (%)', fontsize=12)
ax.set_xlabel('Year of Loan Approval', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('visuals/time_series.png', dpi=300)
plt.close()
print("Saved: visuals/time_series.png")
