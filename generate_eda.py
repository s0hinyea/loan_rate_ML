import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('visuals', exist_ok=True)

df = pd.read_csv('data/df_features.csv')

# ── Chart 1: Industry Risk ────────────────────────────────────────────────────
print("Generating industry risk chart...")
industry_risk = df.groupby('Sector')['default'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
industry_risk.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Default Rate by Industry Sector', fontsize=16, fontweight='bold')
ax.set_xlabel('Industry Sector Code', fontsize=12)
ax.set_ylabel('Default Rate', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visuals/industry_risk.png', dpi=300)
plt.close()
print("Saved: visuals/industry_risk.png")

# ── Chart 2: State Risk ───────────────────────────────────────────────────────
print("Generating state risk chart...")
state_risk = df.groupby('State')['default'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(18, 6))
state_risk.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Default Rate by State — Geographic Bias Signal', fontsize=16, fontweight='bold')
ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Default Rate', fontsize=12)
ax.tick_params(axis='x', rotation=90)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visuals/state_risk.png', dpi=300)
plt.close()
print("Saved: visuals/state_risk.png")

# ── Chart 3: Loan Size Distribution ──────────────────────────────────────────
print("Generating loan distribution chart...")
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['DisbursementGross'], bins=80, color='steelblue', edgecolor='none', alpha=0.85)
ax.set_xscale('log')
ax.set_title('Loan Size Distribution (Log Scale)', fontsize=16, fontweight='bold')
ax.set_xlabel('Loan Amount ($)', fontsize=12)
ax.set_ylabel('Number of Loans', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visuals/loan_distribution.png', dpi=300)
plt.close()
print("Saved: visuals/loan_distribution.png")

# ── Chart 4: Correlation Heatmap ──────────────────────────────────────────────
print("Generating correlation heatmap...")
numeric_cols = df.select_dtypes(include='number')
corr = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr,
    ax=ax,
    cmap='coolwarm',
    center=0,
    annot=False,
    linewidths=0.5,
    linecolor='#2a2a2a'
)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visuals/correlation_heatmap.png', dpi=300)
plt.close()
print("Saved: visuals/correlation_heatmap.png")
