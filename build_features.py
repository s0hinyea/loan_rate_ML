import pandas as pd
import numpy as np

def build_features(df):
    """
    Takes in df_clean and returns df with 8 new engineered features.
    Input:  dataframe with 16 columns from Person B
    Output: same dataframe with 24 columns
    """

    # ── Validation first ──────────────────────────────────────
    required_cols = [
        'DisbursementGross', 'CreateJob', 'ApprovalFY',
        'Sector', 'default', 'NewExist', 'SBA_Appv',
        'GrAppv', 'State'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # ── Feature 1 ─────────────────────────────────────────────
    # loan_to_jobs_ratio
    df['loan_to_jobs_ratio'] = df['DisbursementGross'] / (df['CreateJob'] + 1)
    assert df['loan_to_jobs_ratio'].isnull().sum() == 0
    assert not np.isinf(df['loan_to_jobs_ratio']).any()

    # ── Feature 2 ─────────────────────────────────────────────
    # is_recession
    df['is_recession'] = df['ApprovalFY'].isin([2008, 2009]).astype(int)
    assert df['is_recession'].isin([0, 1]).all()

    # ── Feature 3 ─────────────────────────────────────────────
    # industry_default_rate
    df['industry_default_rate'] = df.groupby('Sector')['default'].transform('mean')
    assert df['industry_default_rate'].between(0, 1).all()
    assert df['industry_default_rate'].isnull().sum() == 0

    # ── Feature 4 ─────────────────────────────────────────────
    # is_new_business
    df['is_new_business'] = (df['NewExist'] == 2).astype(int)
    assert df['is_new_business'].isin([0, 1]).all()

    # ── Feature 5 ─────────────────────────────────────────────
    # sba_coverage_ratio
    # Handle GrAppv == 0 to avoid infinity or NaN
    df['sba_coverage_ratio'] = np.where(df['GrAppv'] > 0, df['SBA_Appv'] / df['GrAppv'], 0.0)
    assert df['sba_coverage_ratio'].isnull().sum() == 0
    assert not np.isinf(df['sba_coverage_ratio']).any()

    # ── Feature 6 ─────────────────────────────────────────────
    # state_default_rate
    df['state_default_rate'] = df.groupby('State')['default'].transform('mean')
    assert df['state_default_rate'].between(0, 1).all()
    assert df['state_default_rate'].isnull().sum() == 0

    # ── Feature 7 ─────────────────────────────────────────────
    # loan_vs_industry_avg
    sector_means = df.groupby('Sector')['DisbursementGross'].transform('mean')
    df['loan_vs_industry_avg'] = np.where(sector_means > 0, df['DisbursementGross'] / sector_means, 1.0)
    assert df['loan_vs_industry_avg'].isnull().sum() == 0
    assert not np.isinf(df['loan_vs_industry_avg']).any()

    # ── Feature 8 ─────────────────────────────────────────────
    # disbursement_ratio
    df['disbursement_ratio'] = np.where(df['GrAppv'] > 0, df['DisbursementGross'] / df['GrAppv'], 0.0)
    assert df['disbursement_ratio'].isnull().sum() == 0
    assert not np.isinf(df['disbursement_ratio']).any()

    # ── Priority 1 — State × Sector interaction ───────────────
    # Joint default rate for this exact state + industry combination.
    # More specific than either state or sector alone — fixes the
    # "Data Hole" where sparse combos fall back to safe individual signals.
    df['state_sector_default_rate'] = (
        df.groupby(['State', 'Sector'])['default']
        .transform('mean')
    )
    assert 'state_sector_default_rate' in df.columns
    assert df['state_sector_default_rate'].nunique() > 50
    assert df['state_sector_default_rate'].isnull().sum() == 0

    # ── Priority 4a — Loan size bucket ────────────────────────
    # Makes the small/large loan threshold explicit for the model.
    # $25k and $26k are treated identically as continuous — this
    # encodes the economically meaningful breaks.
    df['loan_size_bucket'] = pd.cut(
        df['DisbursementGross'],
        bins=[0, 25_000, 75_000, 150_000, 500_000, float('inf')],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    assert df['loan_size_bucket'].isin([0, 1, 2, 3, 4]).all()

    # ── Priority 4b — Zero jobs created flag ──────────────────
    # Loans creating zero jobs default at a meaningfully different
    # rate. loan_to_jobs_ratio exists but doesn't isolate this sharply.
    df['zero_jobs_created'] = (df['CreateJob'] == 0).astype(int)
    assert df['zero_jobs_created'].isin([0, 1]).all()

    # Final Validation Block
    new_features = [
        'loan_to_jobs_ratio',
        'is_recession',
        'industry_default_rate',
        'is_new_business',
        'sba_coverage_ratio',
        'state_default_rate',
        'loan_vs_industry_avg',
        'disbursement_ratio',
        'state_sector_default_rate',
        'loan_size_bucket',
        'zero_jobs_created',
    ]

    # Confirm all 8 features exist
    for f in new_features:
        assert f in df.columns, f"Feature missing: {f}"

    # Confirm no nulls in any new feature
    for f in new_features:
        assert df[f].isnull().sum() == 0, f"Nulls found in: {f}"

    print(f"Feature engineering complete.")
    print(f"Output shape: {df.shape}")
    print(f"New features added: {new_features}")
    
    return df

if __name__ == '__main__':
    import os
    if os.path.exists('data/df_clean.csv'):
        df = pd.read_csv('data/df_clean.csv')
        df_features = build_features(df)
        df_features.to_csv('data/df_features.csv', index=False)
        print("Saved to data/df_features.csv")
    else:
        print("data/df_clean.csv not found yet. Ready to run when data arrives!")
