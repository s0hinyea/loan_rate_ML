import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Is My Business Loan Fair?",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS (Inter font, sidebar gradient, glassmorphic cards, emerald button) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar: Deep Navy → Carbon gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #1a1f2e 60%, #1c2133 100%);
}
[data-testid="stSidebar"] * {
    color: #F3F4F6 !important;
}

/* Glassmorphic metric cards */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 15px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.12);
}

/* Emerald green primary button + hover */
[data-testid="stButton"] button[kind="primary"] {
    background-color: #10B981 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: background-color 0.2s ease !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #059669 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load both pkl files once and cache them."""
    try:
        with open('models/xgb_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/xgb_regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        return classifier, regressor
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Run train_models.py and train_regressor.py first.")
        st.stop()

classifier, regressor = load_models()

# ── Load lookup table ─────────────────────────────────────────────────────────
@st.cache_data
def load_lookup_table():
    """Load df_features.csv as the lookup table."""
    try:
        df = pd.read_csv('data/df_features.csv')
        return df
    except FileNotFoundError:
        st.error("data/df_features.csv missing. Run build_features.py first.")
        st.stop()

df_lookup = load_lookup_table()

# Pre-compute O(1) lookup dicts
state_rate_map    = df_lookup.groupby('State')['state_default_rate'].mean().to_dict()
industry_rate_map = df_lookup.groupby('Sector')['industry_default_rate'].mean().to_dict()
industry_avg_loan = df_lookup.groupby('Sector')['DisbursementGross'].mean().to_dict()
state_avg_loan    = df_lookup.groupby('State')['GrAppv'].mean().to_dict()

# ── Rate formula ──────────────────────────────────────────────────────────────
def coverage_to_fair_rate(sba_coverage: float) -> float:
    """
    Fair Rate = Prime Rate (8.5%) + (1.0 - predicted_sba_coverage_ratio) * 10
    High govt backing → low rate.  Low govt backing → high rate.
    """
    sba_coverage = max(0.0, min(1.0, sba_coverage))
    return round(8.5 + (1.0 - sba_coverage) * 10, 1)

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features_for_model(
    state, sector, loan_amount, jobs_created, jobs_retained,
    num_employees, term_months, business_type, urban_rural,
    low_doc, is_franchise, revolving_credit, df_lookup
):
    """
    Converts human-readable sidebar inputs into the exact numeric feature
    vector our XGBoost models were trained on.
    Returns (input_clf, input_reg).
    """
    state_default_rate    = state_rate_map.get(state, df_lookup['state_default_rate'].mean())
    industry_default_rate = industry_rate_map.get(sector, df_lookup['industry_default_rate'].mean())
    industry_avg          = industry_avg_loan.get(sector, loan_amount)

    loan_to_jobs_ratio   = loan_amount / (jobs_created + 1)
    is_recession         = 0   # user is applying now, not in 2008
    is_new_business      = 1 if business_type == "New Business" else 0
    loan_vs_industry_avg = loan_amount / industry_avg if industry_avg > 0 else 1.0
    sba_coverage_ratio   = df_lookup['sba_coverage_ratio'].mean()
    disbursement_ratio   = df_lookup['disbursement_ratio'].mean()

    urban_rural_val = 1 if urban_rural == "Urban" else 2
    low_doc_val     = 1 if low_doc == "LowDoc (Fast Track)" else 0
    franchise_val   = 1 if is_franchise else 0
    rev_line_val    = 1 if revolving_credit else 0

    features = {
        'ApprovalFY':            2024,
        'Term':                  term_months,
        'NoEmp':                 num_employees,
        'NewExist':              2 if is_new_business else 1,
        'CreateJob':             jobs_created,
        'RetainedJob':           jobs_retained,
        'IsFranchise':           franchise_val,
        'UrbanRural':            urban_rural_val,
        'RevLineCr':             rev_line_val,
        'LowDoc':                low_doc_val,
        'DisbursementGross':     loan_amount,
        'loan_to_jobs_ratio':    loan_to_jobs_ratio,
        'is_recession':          is_recession,
        'industry_default_rate': industry_default_rate,
        'is_new_business':       is_new_business,
        'sba_coverage_ratio':    sba_coverage_ratio,
        'state_default_rate':    state_default_rate,
        'loan_vs_industry_avg':  loan_vs_industry_avg,
        'disbursement_ratio':    disbursement_ratio,
    }

    input_clf = pd.DataFrame([features])
    input_reg = input_clf.drop(columns=['sba_coverage_ratio', 'disbursement_ratio', 'DisbursementGross'])

    return input_clf, input_reg

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists('app/logo.png'):
        st.image('app/logo.png', use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.title("Your Business Profile")
    st.caption("Fill in your details. We'll tell you if your loan is fair.")
    st.divider()

    # State
    all_states = sorted(df_lookup['State'].dropna().unique().tolist())
    selected_state = st.selectbox(
        "📍 State",
        options=all_states,
        index=all_states.index('NY') if 'NY' in all_states else 0,
        help="The state where your business is located"
    )

    # Industry
    SECTOR_LABELS = {
        '11': 'Agriculture & Forestry',
        '21': 'Mining & Oil',
        '22': 'Utilities',
        '23': 'Construction',
        '31': 'Manufacturing',
        '32': 'Manufacturing',
        '33': 'Manufacturing',
        '42': 'Wholesale Trade',
        '44': 'Retail Trade',
        '45': 'Retail Trade',
        '48': 'Transportation',
        '49': 'Transportation',
        '51': 'Information',
        '52': 'Finance & Insurance',
        '53': 'Real Estate',
        '54': 'Professional Services',
        '55': 'Management',
        '56': 'Administrative Services',
        '61': 'Educational Services',
        '62': 'Healthcare',
        '71': 'Arts & Entertainment',
        '72': 'Accommodation & Food (Restaurants)',
        '81': 'Other Services',
        '92': 'Public Administration',
    }
    available_sectors = sorted(df_lookup['Sector'].dropna().unique().tolist())
    sector_display    = {s: SECTOR_LABELS.get(s, f'Sector {s}') for s in available_sectors}

    selected_sector_label = st.selectbox(
        "🏢 Industry",
        options=list(sector_display.values()),
        help="The type of business you operate"
    )
    selected_sector = [k for k, v in sector_display.items()
                       if v == selected_sector_label][0]

    st.divider()

    loan_amount = st.number_input(
        "💵 Loan Amount ($)",
        min_value=5_000,
        max_value=5_000_000,
        value=150_000,
        step=5_000,
        help="The total amount you are requesting"
    )
    jobs_created = st.number_input(
        "👷 Jobs This Loan Will Create",
        min_value=0, max_value=500, value=5,
        help="How many new jobs will this loan help create?"
    )
    jobs_retained = st.number_input(
        "🤝 Jobs Retained",
        min_value=0, max_value=500, value=3,
        help="How many existing jobs will this loan help keep?"
    )
    num_employees = st.number_input(
        "👥 Current Number of Employees",
        min_value=1, max_value=1000, value=10
    )
    term_months = st.slider(
        "📅 Loan Term (Months)",
        min_value=12, max_value=300, value=84, step=12,
        help="How long you want to repay the loan. 84 months = 7 years."
    )
    business_type = st.radio(
        "🏗️ Business Status",
        options=["Existing Business", "New Business"],
        help="New businesses carry higher default risk"
    )
    urban_rural = st.radio(
        "🌆 Location Type",
        options=["Urban", "Rural"]
    )
    low_doc = st.radio(
        "📋 Loan Program",
        options=["Standard Application", "LowDoc (Fast Track)"],
        help="LowDoc loans have less paperwork but may carry more risk"
    )
    is_franchise = st.checkbox(
        "🔖 This is a Franchise",
        help="e.g. McDonald's, Subway, etc."
    )
    revolving_credit = st.checkbox(
        "🔄 Revolving Line of Credit",
        help="A credit line you can draw from repeatedly"
    )

# ── Main body ─────────────────────────────────────────────────────────────────
st.title("💰 Is My Business Loan Rate Fair?")
st.caption(
    "Built on 900,000 real SBA loans (1987–2014). "
    "Enter your business profile in the sidebar and click Analyze."
)
st.divider()

if st.button("🔍 Analyze My Loan", type="primary", use_container_width=True):

    with st.spinner("Analyzing your loan profile..."):
        input_clf, input_reg = engineer_features_for_model(
            selected_state, selected_sector, loan_amount, jobs_created,
            jobs_retained, num_employees, term_months, business_type,
            urban_rural, low_doc, is_franchise, revolving_credit, df_lookup
        )
        risk_prob    = classifier.predict_proba(input_clf)[0][1]
        sba_coverage = float(regressor.predict(input_reg)[0])
        fair_rate    = coverage_to_fair_rate(sba_coverage)

    # Historical comparison for this state + sector
    similar = df_lookup[
        (df_lookup['State']  == selected_state) &
        (df_lookup['Sector'] == selected_sector)
    ]
    hist_avg_loan  = similar['GrAppv'].mean()            if len(similar) > 0 else loan_amount
    hist_default   = similar['default'].mean()            if len(similar) > 0 else 0
    hist_sba       = similar['sba_coverage_ratio'].mean() if len(similar) > 0 else sba_coverage
    hist_fair_rate = coverage_to_fair_rate(hist_sba)
    rate_gap       = fair_rate - hist_fair_rate
    dollar_gap     = loan_amount * (rate_gap / 100) * (term_months / 12)

    # Metrics row
    st.subheader("Your Results")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Default Risk",
        value=f"{risk_prob * 100:.1f}%",
        delta=f"{(risk_prob - hist_default) * 100:+.1f}% vs your area avg",
        delta_color="inverse"
    )
    col2.metric(
        label="Fair Interest Rate",
        value=f"{fair_rate:.1f}%",
        delta=f"{rate_gap:+.1f}% vs similar businesses",
        delta_color="inverse"
    )
    col3.metric(
        label="Govt Should Back",
        value=f"{sba_coverage * 100:.1f}%",
        help="How much of your loan the government should guarantee based on your risk profile"
    )
    col4.metric(
        label="Avg Loan in Your Area",
        value=f"${hist_avg_loan:,.0f}",
        delta=f"{((loan_amount - hist_avg_loan) / hist_avg_loan) * 100:+.1f}% vs your request",
        delta_color="off"
    )

    # Narrative
    st.divider()
    st.subheader("What This Means For You")

    risk_label = (
        "🟢 Low Risk"    if risk_prob < 0.20 else
        "🟡 Medium Risk" if risk_prob < 0.40 else
        "🔴 High Risk"
    )

    st.info(
        f"**{risk_label}** — Based on 900,000 historical loans, a "
        f"**{sector_display[selected_sector]}** business in **{selected_state}** "
        f"requesting **${loan_amount:,}** has a "
        f"**{risk_prob * 100:.1f}%** chance of defaulting.\n\n"
        f"The government should back **{sba_coverage * 100:.1f}%** of your loan, "
        f"which maps to a fair rate of **{fair_rate:.1f}%**.\n\n"
        f"Similar businesses in your state and industry historically received "
        f"a rate implying **{hist_fair_rate:.1f}%**. "
        f"That gap of **{rate_gap:+.1f} percentage points** costs you roughly "
        f"**${abs(dollar_gap):,.0f}** over the life of your loan."
    )

    st.progress(
        value=float(risk_prob),
        text=f"Default Risk: {risk_prob * 100:.1f}%"
    )

    st.balloons()

# ── Evidence panels ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 The Evidence Behind This Model")

core_col1, core_col2 = st.columns(2)

with core_col1:
    st.markdown("**What drives default risk? (SHAP)**")
    if os.path.exists('visuals/shap_summary.png'):
        st.image('visuals/shap_summary.png', use_container_width=True)
    else:
        st.warning(
            "⏳ SHAP summary chart not yet generated.\n\n"
            "Run generate_visuals.py to produce visuals/shap_summary.png"
        )

with core_col2:
    st.markdown("**Default rates over time — the 2008 effect**")
    if os.path.exists('visuals/time_series.png'):
        st.image('visuals/time_series.png', use_container_width=True)
    else:
        st.warning(
            "⏳ Time series chart not yet generated.\n\n"
            "Run generate_visuals.py to produce visuals/time_series.png"
        )

# EDA placeholders
st.divider()
st.subheader("🗺️ Geographic & Industry Analysis")
st.caption(
    "These visuals are in progress. "
    "They will appear automatically once the EDA charts are generated."
)

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.markdown("**Default Rate by Industry**")
    if os.path.exists('visuals/industry_risk.png'):
        st.image('visuals/industry_risk.png', use_container_width=True)
    else:
        st.info("📌 Coming soon — industry_risk.png (Person B EDA task)")

with eda_col2:
    st.markdown("**Default Rate by State**")
    if os.path.exists('visuals/state_risk.png'):
        st.image('visuals/state_risk.png', use_container_width=True)
    else:
        st.info("📌 Coming soon — state_risk.png (Person B EDA task)")

eda_col3, eda_col4 = st.columns(2)

with eda_col3:
    st.markdown("**Loan Size Distribution**")
    if os.path.exists('visuals/loan_distribution.png'):
        st.image('visuals/loan_distribution.png', use_container_width=True)
    else:
        st.info("📌 Coming soon — loan_distribution.png (Person B EDA task)")

with eda_col4:
    st.markdown("**Feature Correlation Heatmap**")
    if os.path.exists('visuals/correlation_heatmap.png'):
        st.image('visuals/correlation_heatmap.png', use_container_width=True)
    else:
        st.info("📌 Coming soon — correlation_heatmap.png (Person B EDA task)")

# Choropleth
st.divider()
st.subheader("🗺️ Interactive US Default Rate Map")
st.caption("Advanced feature — geographic bias visualized across all 50 states.")

if os.path.exists('visuals/choropleth.html'):
    with open('visuals/choropleth.html', 'r') as f:
        choropleth_html = f.read()
    st.components.v1.html(choropleth_html, height=500)
else:
    st.info(
        "📌 Choropleth map coming soon.\n\n"
        "Person B: run choropleth.py using Plotly Express and save "
        "the figure as visuals/choropleth.html using fig.write_html()"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built at AI Community Datathon 2026 · Stony Brook University · Finance Track  |  "
    "Data: SBA National Loan Dataset 1987–2014 · 900,000+ real loans  |  "
    "Model: XGBoost Classifier (F1: 0.86) + XGBoost Regressor"
)
