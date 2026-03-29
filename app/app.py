import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from build_features import build_features

st.set_page_config(
    page_title="FairRate — Is My Rate Fair?",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ─────────────────────────────────────────────────────────────
# Palette: Mint (#3DDAB4) primary · Navy (#080E1A) bg · Coral (#FF6B6B) danger
# Amber (#FBBF24) warning · Cyan (#06B6D4) info · Slate (#1E293B) surfaces
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080E1A;
    color: #E2E8F0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0D1B2A 0%, #0F2137 50%, #0A1628 100%);
    border-right: 1px solid rgba(61, 218, 180, 0.15);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1 {
    color: #3DDAB4 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(61, 218, 180, 0.06) !important;
    border: 1px solid rgba(61, 218, 180, 0.2) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input {
    background-color: rgba(61, 218, 180, 0.06) !important;
    border: 1px solid rgba(61, 218, 180, 0.2) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background-color: #3DDAB4 !important;
}

/* ── Main background ── */
.main .block-container {
    background-color: #080E1A;
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(61,218,180,0.08) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid rgba(61, 218, 180, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(61, 218, 180, 0.5);
    transform: translateY(-2px);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #3DDAB4 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Analyze button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #3DDAB4 0%, #06B6D4 100%) !important;
    color: #080E1A !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.04em !important;
    padding: 14px 0 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(61, 218, 180, 0.3) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2EC9A3 0%, #0EA5C9 100%) !important;
    box-shadow: 0 6px 28px rgba(61, 218, 180, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Dividers ── */
hr {
    border-color: rgba(61, 218, 180, 0.12) !important;
    margin: 2rem 0 !important;
}

/* ── Info / warning boxes ── */
[data-testid="stAlert"] {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    border-left: 3px solid #3DDAB4 !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #3DDAB4, #06B6D4) !important;
    border-radius: 999px !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(61, 218, 180, 0.1) !important;
    border-radius: 999px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(61, 218, 180, 0.15) !important;
    border-radius: 10px !important;
    background: rgba(30, 41, 59, 0.4) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #3DDAB4 !important; }

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #475569 !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open('models/xgb_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/xgb_regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        threshold = 0.5  # fallback if not yet generated
        if os.path.exists('models/best_threshold.pkl'):
            with open('models/best_threshold.pkl', 'rb') as f:
                threshold = pickle.load(f)
        return classifier, regressor, threshold
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Run train_models.py and train_regressor.py first.")
        st.stop()

classifier, regressor, RISK_THRESHOLD = load_models()

@st.cache_data
def load_lookup_table():
    feature_path = 'data/df_features.csv'
    clean_path = 'data/df_clean.csv'

    if os.path.exists(feature_path):
        return pd.read_csv(feature_path)

    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        df_features = build_features(df_clean)
        try:
            df_features.to_csv(feature_path, index=False)
        except OSError:
            # Some deploy targets may have read-only app directories.
            pass
        return df_features

    st.error(
        "Missing both data/df_features.csv and data/df_clean.csv. "
        "Deploy one of those files so the app can build its lookup table."
    )
    st.stop()

df_lookup = load_lookup_table()

state_rate_map         = df_lookup.groupby('State')['state_default_rate'].mean().to_dict()
industry_rate_map      = df_lookup.groupby('Sector')['industry_default_rate'].mean().to_dict()
industry_avg_loan      = df_lookup.groupby('Sector')['DisbursementGross'].mean().to_dict()
state_sector_rate_map  = df_lookup.groupby(['State','Sector'])['state_sector_default_rate'].mean().to_dict() if 'state_sector_default_rate' in df_lookup.columns else {}
combo_count_map        = {
    (state, str(int(sector))): int(count)
    for (state, sector), count in df_lookup.groupby(['State', 'Sector']).size().items()
}
RATE_TOLERANCE = 0.5

def coverage_to_fair_rate(sba_coverage: float) -> float:
    sba_coverage = max(0.0, min(1.0, sba_coverage))
    return round(8.5 + (1.0 - sba_coverage) * 10, 1)

def format_loan_count(count: int) -> str:
    return f"{count:,} loan" if count == 1 else f"{count:,} loans"

def get_support_info(state: str, sector: str) -> dict:
    count = combo_count_map.get((state, str(sector)), 0)
    if count < 10:
        tier = "Very Limited Support"
        detail = "Low confidence"
        limited = True
        color = "#FBBF24"
    elif count < 30:
        tier = "Limited Support"
        detail = "Lower confidence"
        limited = True
        color = "#F59E0B"
    else:
        tier = "Strong Support"
        detail = "Higher confidence"
        limited = False
        color = "#3DDAB4"

    return {
        'count': count,
        'tier': tier,
        'detail': detail,
        'limited': limited,
        'color': color,
        'compact_label': f"Historical support: {format_loan_count(count)} · {detail}",
    }

def engineer_features_for_model(
    state, sector, loan_amount, jobs_created, jobs_retained,
    num_employees, term_months, business_type, urban_rural,
    low_doc, is_franchise, revolving_credit, df_lookup
):
    # Ensure sector is int for lookup in maps (which used df columns as keys)
    sector_int = int(sector)
    state_default_rate    = state_rate_map.get(state, df_lookup['state_default_rate'].mean())
    industry_default_rate = industry_rate_map.get(sector_int, df_lookup['industry_default_rate'].mean())
    industry_avg          = industry_avg_loan.get(sector_int, loan_amount)

    loan_to_jobs_ratio        = loan_amount / (jobs_created + 1)
    is_new_business           = 1 if business_type == "New Business" else 0
    loan_vs_industry_avg      = loan_amount / industry_avg if industry_avg > 0 else 1.0
    sba_coverage_ratio        = df_lookup['sba_coverage_ratio'].mean()
    disbursement_ratio        = df_lookup['disbursement_ratio'].mean()
    gr_appv                   = loan_amount
    sba_appv                  = loan_amount * sba_coverage_ratio
    state_sector_default_rate = state_sector_rate_map.get(
        (state, sector_int),
        state_rate_map.get(state, df_lookup['state_default_rate'].mean())
    )
    loan_size_bucket = pd.cut(
        pd.Series([loan_amount]),
        bins=[0, 25_000, 75_000, 150_000, 500_000, float('inf')],
        labels=[0, 1, 2, 3, 4]
    ).astype(int).iloc[0]
    zero_jobs_created = 1 if jobs_created == 0 else 0

    features = {
        'ApprovalFY':            2024,
        'Term':                  term_months,
        'NoEmp':                 num_employees,
        'NewExist':              2 if is_new_business else 1,
        'CreateJob':             jobs_created,
        'RetainedJob':           jobs_retained,
        'IsFranchise':           1 if is_franchise else 0,
        'UrbanRural':            1 if urban_rural == "Urban" else 2,
        'RevLineCr':             1 if revolving_credit else 0,
        'LowDoc':                1 if low_doc == "LowDoc (Fast Track)" else 0,
        'DisbursementGross':     loan_amount,
        'GrAppv':                gr_appv,
        'SBA_Appv':              sba_appv,
        'loan_to_jobs_ratio':    loan_to_jobs_ratio,
        'is_recession':          0,
        'industry_default_rate': industry_default_rate,
        'is_new_business':       is_new_business,
        'sba_coverage_ratio':    sba_coverage_ratio,
        'state_default_rate':    state_default_rate,
        'loan_vs_industry_avg':       loan_vs_industry_avg,
        'disbursement_ratio':         disbursement_ratio,
        'state_sector_default_rate':  state_sector_default_rate,
        'loan_size_bucket':           loan_size_bucket,
        'zero_jobs_created':          zero_jobs_created,
    }

    # CalibratedClassifierCV wraps the base estimator — get feature names safely
    clf_features = (
        classifier.feature_names_in_
        if hasattr(classifier, 'feature_names_in_')
        else classifier.estimator.feature_names_in_
    )
    input_clf = pd.DataFrame([features])[clf_features]
    input_reg = input_clf.drop(columns=['sba_coverage_ratio', 'GrAppv', 'SBA_Appv'], errors='ignore')
    input_reg = input_reg[regressor.feature_names_in_]
    return input_clf, input_reg

# ── Sector labels ─────────────────────────────────────────────────────────────
SECTOR_LABELS = {
    '11': 'Agriculture & Forestry', '21': 'Mining & Oil', '22': 'Utilities',
    '23': 'Construction', '31': 'Manufacturing', '32': 'Manufacturing',
    '33': 'Manufacturing', '42': 'Wholesale Trade', '44': 'Retail Trade',
    '45': 'Retail Trade', '48': 'Transportation', '49': 'Transportation',
    '51': 'Information', '52': 'Finance & Insurance', '53': 'Real Estate',
    '54': 'Professional Services', '55': 'Management',
    '56': 'Administrative Services', '61': 'Educational Services',
    '62': 'Healthcare', '71': 'Arts & Entertainment',
    '72': 'Accommodation & Food (Restaurants)',
    '81': 'Other Services', '92': 'Public Administration',
}

all_states = sorted(df_lookup['State'].dropna().astype(str).unique().tolist())
available_sectors = sorted(
    df_lookup['Sector'].dropna().astype(int).astype(str).unique().tolist(),
    key=int
)
sector_display = {s: SECTOR_LABELS.get(s, f'Sector {s}') for s in available_sectors}

WIDGET_DEFAULTS = {
    'selected_state': 'NY' if 'NY' in all_states else all_states[0],
    'selected_sector': '62' if '62' in available_sectors else available_sectors[0],
    'loan_amount': 150_000,
    'term_months': 84,
    'quoted_rate': 8.9,
    'jobs_created': 5,
    'jobs_retained': 3,
    'num_employees': 10,
    'business_type': 'Existing Business',
    'urban_rural': 'Urban',
    'low_doc': 'Standard Application',
    'is_franchise': False,
    'revolving_credit': False,
}

DEMO_PRESETS = {
    'Low Risk Baseline': {
        'selected_state': 'ND',
        'selected_sector': '62',
        'loan_amount': 150_000,
        'term_months': 84,
        'quoted_rate': 13.4,
        'jobs_created': 5,
        'jobs_retained': 5,
        'num_employees': 10,
        'business_type': 'Existing Business',
        'urban_rural': 'Urban',
        'low_doc': 'Standard Application',
        'is_franchise': False,
        'revolving_credit': False,
    },
    'Medium Risk Overcharge': {
        'selected_state': 'CA',
        'selected_sector': '44',
        'loan_amount': 20_000,
        'term_months': 48,
        'quoted_rate': 15.0,
        'jobs_created': 1,
        'jobs_retained': 0,
        'num_employees': 2,
        'business_type': 'New Business',
        'urban_rural': 'Urban',
        'low_doc': 'Standard Application',
        'is_franchise': False,
        'revolving_credit': False,
    },
    'High Risk Overcharge': {
        'selected_state': 'CA',
        'selected_sector': '44',
        'loan_amount': 20_000,
        'term_months': 30,
        'quoted_rate': 16.5,
        'jobs_created': 2,
        'jobs_retained': 1,
        'num_employees': 5,
        'business_type': 'Existing Business',
        'urban_rural': 'Urban',
        'low_doc': 'Standard Application',
        'is_franchise': False,
        'revolving_credit': False,
    },
}

def ensure_widget_defaults():
    for key, value in WIDGET_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault('active_demo_preset', None)

def load_demo_preset(name: str):
    for key, value in DEMO_PRESETS[name].items():
        st.session_state[key] = value
    st.session_state['active_demo_preset'] = name

ensure_widget_defaults()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists('app/logo.png'):
        st.image('app/logo.png', use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:0.25rem'>
        <span style='color:#06B6D4;font-size:0.65rem;font-weight:700;
        letter-spacing:0.12em;text-transform:uppercase'>
        ◆ Quick Demo Presets
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Load a stable live demo example, then tweak any field you want.")
    for preset_name in DEMO_PRESETS:
        st.button(
            preset_name,
            key=f"preset_{preset_name.lower().replace(' ', '_')}",
            on_click=load_demo_preset,
            args=(preset_name,),
            use_container_width=True
        )
    if st.session_state.get('active_demo_preset'):
        st.caption(
            f"Loaded preset: {st.session_state['active_demo_preset']}. "
            "All inputs remain editable."
        )
    st.divider()

    st.markdown("""
    <div style='margin-bottom:0.25rem'>
        <span style='color:#3DDAB4;font-size:0.65rem;font-weight:700;
        letter-spacing:0.12em;text-transform:uppercase'>
        ◆ Business Profile
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Fill in your details below.")
    st.divider()

    selected_state = st.selectbox(
        "State",
        options=all_states,
        key='selected_state',
        help="The state where your business is located"
    )

    selected_sector = st.selectbox(
        "Industry",
        options=available_sectors,
        key='selected_sector',
        format_func=lambda code: sector_display.get(str(code), f'Sector {code}'),
        help="The type of business you operate"
    )

    with st.expander("Industry Code Reference"):
        ref_df = pd.DataFrame({
            "Code": available_sectors,
            "Industry": [sector_display[code] for code in available_sectors]
        })
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

    st.divider()

    loan_amount = st.number_input(
        "Loan Amount ($)", min_value=5_000, max_value=5_000_000,
        step=5_000, key='loan_amount', help="Total amount you are requesting"
    )
    term_months = st.slider(
        "Loan Term (Months)", min_value=6, max_value=300,
        step=6, key='term_months', help="84 months = 7 years"
    )
    quoted_rate = st.number_input(
        "Bank Quoted Rate (%)", min_value=0.0, max_value=40.0,
        step=0.1, key='quoted_rate',
        help="The annual interest rate your bank offered for this loan"
    )

    st.divider()

    c1, c2 = st.columns(2)
    jobs_created  = c1.number_input("Jobs Created",  min_value=0, max_value=500, key='jobs_created')
    jobs_retained = c2.number_input("Jobs Retained", min_value=0, max_value=500, key='jobs_retained')
    num_employees = st.number_input("Current Employees", min_value=1, max_value=1000, key='num_employees')

    st.divider()

    business_type    = st.radio("Business Status", ["Existing Business", "New Business"], key='business_type')
    urban_rural      = st.radio("Location Type",   ["Urban", "Rural"], key='urban_rural')
    low_doc          = st.radio("Loan Program",     ["Standard Application", "LowDoc (Fast Track)"], key='low_doc')
    is_franchise     = st.checkbox("Franchise Business", help="e.g. McDonald's, Subway", key='is_franchise')
    revolving_credit = st.checkbox("Revolving Line of Credit", key='revolving_credit')

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2.5rem 0 1.5rem 0;'>
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;'>
        <div style='width:4px; height:40px; background:linear-gradient(180deg,#3DDAB4,#06B6D4);
             border-radius:2px;'></div>
        <div>
            <div style='color:#475569; font-size:0.7rem; font-weight:600;
                 letter-spacing:0.14em; text-transform:uppercase; margin-bottom:4px;'>
                SBA Loan Intelligence 
            </div>
            <h1 style='margin:0; font-size:2rem; font-weight:700; color:#F1F5F9;
                 letter-spacing:-0.02em; line-height:1.1;'>
                Is My Business Loan Rate <span style='color:#3DDAB4;'>Fair?</span>
            </h1>
        </div>
    </div>
    <p style='color:#64748B; font-size:0.88rem; margin: 0.75rem 0 0 16px; max-width:600px;'>
        Enter your business profile in the sidebar. Our models will estimate your default risk,
        calculate a fair rate, and compare it directly against your bank's quote.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Analyze Button ────────────────────────────────────────────────────────────
if st.button("Analyze My Loan →", type="primary", use_container_width=True):

    with st.spinner("Running models..."):
        input_clf, input_reg = engineer_features_for_model(
            selected_state, selected_sector, loan_amount, jobs_created,
            jobs_retained, num_employees, term_months, business_type,
            urban_rural, low_doc, is_franchise, revolving_credit, df_lookup
        )
        risk_prob    = classifier.predict_proba(input_clf)[0][1]
        sba_coverage = float(regressor.predict(input_reg)[0])
        fair_rate    = coverage_to_fair_rate(sba_coverage)

    selected_sector_int = int(selected_sector)
    similar = df_lookup[
        (df_lookup['State'] == selected_state) &
        (df_lookup['Sector'] == selected_sector_int)
    ]
    support_info = get_support_info(selected_state, selected_sector)
    hist_avg_loan  = similar['GrAppv'].mean()            if len(similar) > 0 else loan_amount
    hist_default   = similar['default'].mean()            if len(similar) > 0 else 0
    hist_sba       = similar['sba_coverage_ratio'].mean() if len(similar) > 0 else sba_coverage
    hist_fair_rate = coverage_to_fair_rate(hist_sba)
    benchmark_gap  = fair_rate - hist_fair_rate
    quote_gap      = quoted_rate - fair_rate
    term_years     = term_months / 12
    quote_dollar_gap = loan_amount * (quote_gap / 100) * term_years
    avg_loan_delta = ((loan_amount - hist_avg_loan) / hist_avg_loan) * 100 if hist_avg_loan else 0.0
    if support_info['count'] == 0:
        risk_delta = "No exact local avg yet"
    elif support_info['limited']:
        risk_delta = f"{(risk_prob - hist_default)*100:+.1f}% vs thin local avg"
    else:
        risk_delta = f"{(risk_prob - hist_default)*100:+.1f}% vs area avg"
    if support_info['count'] == 0:
        benchmark_delta = "No exact local benchmark yet"
    elif support_info['limited']:
        benchmark_delta = f"{benchmark_gap:+.1f}pp vs thin local benchmark"
    else:
        benchmark_delta = f"{benchmark_gap:+.1f}pp vs similar businesses"

    if quote_gap > RATE_TOLERANCE:
        quote_color, quote_label, quote_icon = "#FF6B6B", "Potential Overcharge", "▲"
        quote_metric_label = "Potential Overcharge"
        quote_metric_value = f"${abs(quote_dollar_gap):,.0f}"
        quote_metric_delta = f"{quote_gap:+.1f}pp vs fair rate"
        quote_summary = (
            f"The bank quoted <strong style='color:#F1F5F9;'>{quoted_rate:.1f}%</strong>, which is "
            f"<strong style='color:{quote_color};'>{quote_gap:+.1f} percentage points</strong> above our fair-rate estimate."
        )
        quote_impact = (
            f"That gap may cost roughly <strong style='color:{quote_color};'>${abs(quote_dollar_gap):,.0f}</strong> "
            f"over <strong style='color:#F1F5F9;'>{term_years:.1f} years</strong> using simple interest."
        )
    elif quote_gap < -RATE_TOLERANCE:
        quote_color, quote_label, quote_icon = "#3DDAB4", "Below Fair Rate", "▼"
        quote_metric_label = "Potential Savings"
        quote_metric_value = f"${abs(quote_dollar_gap):,.0f}"
        quote_metric_delta = f"{quote_gap:+.1f}pp vs fair rate"
        quote_summary = (
            f"The bank quoted <strong style='color:#F1F5F9;'>{quoted_rate:.1f}%</strong>, which is "
            f"<strong style='color:{quote_color};'>{quote_gap:+.1f} percentage points</strong> below our fair-rate estimate."
        )
        quote_impact = (
            f"That looks favorable by roughly <strong style='color:{quote_color};'>${abs(quote_dollar_gap):,.0f}</strong> "
            f"over <strong style='color:#F1F5F9;'>{term_years:.1f} years</strong> using simple interest."
        )
    else:
        quote_color, quote_label, quote_icon = "#06B6D4", "Within Fair Range", "●"
        quote_metric_label = "Quote Fairness"
        quote_metric_value = "Within Range"
        quote_metric_delta = f"{quote_gap:+.1f}pp vs fair rate"
        quote_summary = (
            f"The bank quoted <strong style='color:#F1F5F9;'>{quoted_rate:.1f}%</strong>, which lands within our "
            f"<strong style='color:{quote_color};'>±{RATE_TOLERANCE:.1f}pp fairness band</strong> around the model estimate."
        )
        quote_impact = (
            f"The quote is close enough to the modeled fair rate that we would not flag a meaningful overcharge."
        )

    # Risk color — thresholds derived from optimal threshold tuning
    low_thresh = RISK_THRESHOLD * 0.5
    if risk_prob < low_thresh:
        risk_color, risk_label, risk_icon = "#3DDAB4", "Low Risk", "●"
    elif risk_prob < RISK_THRESHOLD:
        risk_color, risk_label, risk_icon = "#FBBF24", "Medium Risk", "●"
    else:
        risk_color, risk_label, risk_icon = "#FF6B6B", "High Risk", "●"

    if support_info['limited']:
        st.warning(
            f"{support_info['tier']}: only {format_loan_count(support_info['count'])} "
            f"match this exact {selected_state} + {sector_display[selected_sector]} combination. "
            "We still show the prediction, but confidence is lower and this segment may "
            "behave less smoothly than denser parts of the dataset."
        )

    # ── Section label ──
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:1.25rem;margin-top:0.5rem;'>
        <div style='width:3px;height:20px;background:#3DDAB4;border-radius:2px;'></div>
        <span style='color:#94A3B8;font-size:0.7rem;font-weight:600;
             letter-spacing:0.12em;text-transform:uppercase;'>Your Results</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:inline-block;margin-top:-0.35rem;margin-bottom:1rem;
         padding:0.4rem 0.7rem;border-radius:999px;
         background:rgba(30,41,59,0.6);
         border:1px solid {support_info['color']};
         color:#CBD5E1;font-size:0.74rem;font-weight:600;'>
        {support_info['compact_label']}
    </div>
    """, unsafe_allow_html=True)

    # ── 4 Metric cards ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Default Risk",         f"{risk_prob * 100:.1f}%",
                risk_delta,                                             delta_color="inverse")
    col2.metric("Fair Interest Rate",   f"{fair_rate:.1f}%",
                benchmark_delta,                                        delta_color="off")
    col3.metric("Bank Quoted Rate",     f"{quoted_rate:.1f}%",
                quote_metric_delta,                                    delta_color="inverse")
    col4.metric(quote_metric_label,     quote_metric_value,
                f"SBA backs {sba_coverage*100:.1f}% · avg loan {avg_loan_delta:+.1f}%",
                delta_color="off")

    # ── Risk gauge ──
    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.progress(float(min(risk_prob, 1.0)),
                text=f"{risk_icon} {risk_label}  ·  {risk_prob*100:.1f}% default probability")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Narrative card ──
    if support_info['count'] == 0:
        evidence_text = (
            f"The broader SBA dataset supports this estimate, but there are "
            f"<strong style='color:#F1F5F9;'>no exact historical loans</strong> in the "
            f"<strong style='color:#F1F5F9;'>{selected_state} · {sector_display[selected_sector]}</strong> pocket. "
            "Treat the local comparison as directional rather than precise. "
            f"With that caveat, a business requesting <strong style='color:#F1F5F9;'>${loan_amount:,}</strong> carries a "
            f"<strong style='color:{risk_color};'>{risk_prob*100:.1f}%</strong> default probability."
        )
        benchmark_text = (
            f"The government should back <strong style='color:#3DDAB4;'>{sba_coverage*100:.1f}%</strong> "
            f"of your loan, mapping to a fair rate of <strong style='color:#3DDAB4;'>{fair_rate:.1f}%</strong>. "
            "Because there is no exact local benchmark, the app is leaning more on broader state, sector, term, "
            "and loan-size patterns than on dense local history."
        )
    elif support_info['limited']:
        evidence_text = (
            f"The broader SBA dataset supports this estimate, but only "
            f"<strong style='color:#F1F5F9;'>{format_loan_count(support_info['count'])}</strong> directly match the "
            f"<strong style='color:#F1F5F9;'>{selected_state} · {sector_display[selected_sector]}</strong> pocket. "
            "Treat the local comparison as directional rather than precise. "
            f"With that caveat, a business requesting <strong style='color:#F1F5F9;'>${loan_amount:,}</strong> carries a "
            f"<strong style='color:{risk_color};'>{risk_prob*100:.1f}%</strong> default probability."
        )
        benchmark_text = (
            f"The government should back <strong style='color:#3DDAB4;'>{sba_coverage*100:.1f}%</strong> "
            f"of your loan, mapping to a fair rate of <strong style='color:#3DDAB4;'>{fair_rate:.1f}%</strong>. "
            f"The thin local benchmark implies <strong style='color:#F1F5F9;'>{hist_fair_rate:.1f}%</strong>, "
            f"so the <strong style='color:#F1F5F9;'>{benchmark_gap:+.1f}pp</strong> gap should be read as a rough signal, "
            "not a precise local market quote."
        )
    else:
        evidence_text = (
            f"Based on <strong style='color:#F1F5F9;'>900,000 historical loans</strong>, including "
            f"<strong style='color:#F1F5F9;'>{format_loan_count(support_info['count'])}</strong> in this exact "
            f"<strong style='color:#F1F5F9;'>{selected_state} · {sector_display[selected_sector]}</strong> pocket, "
            f"a business requesting <strong style='color:#F1F5F9;'>${loan_amount:,}</strong> carries a "
            f"<strong style='color:{risk_color};'>{risk_prob*100:.1f}%</strong> default probability."
        )
        benchmark_text = (
            f"The government should back <strong style='color:#3DDAB4;'>{sba_coverage*100:.1f}%</strong> "
            f"of your loan, mapping to a fair rate of <strong style='color:#3DDAB4;'>{fair_rate:.1f}%</strong>. "
            f"Similar businesses historically imply <strong style='color:#F1F5F9;'>{hist_fair_rate:.1f}%</strong>, "
            f"so your profile sits <strong style='color:#F1F5F9;'>{benchmark_gap:+.1f}pp</strong> from that benchmark."
        )
    st.markdown(f"""
    <div style='margin-top:1.5rem; padding:1.5rem 1.75rem;
         background:linear-gradient(135deg,rgba(30,41,59,0.9) 0%,rgba(15,33,55,0.9) 100%);
         border:1px solid rgba(61,218,180,0.2); border-radius:16px;
         border-left: 4px solid {risk_color};'>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:0.75rem;'>
            <span style='color:{risk_color};font-weight:700;font-size:0.95rem;'>
                {risk_icon} {risk_label}
            </span>
            <span style='color:#475569;font-size:0.8rem;'>·</span>
            <span style='color:{quote_color};font-weight:700;font-size:0.85rem;'>
                {quote_icon} {quote_label}
            </span>
            <span style='color:#475569;font-size:0.8rem;'>·</span>
            <span style='color:#64748B;font-size:0.8rem;'>
                {sector_display[selected_sector]} · {selected_state}
            </span>
        </div>
        <p style='color:#CBD5E1;font-size:0.9rem;line-height:1.65;margin:0 0 0.75rem 0;'>
            {evidence_text}
        </p>
        <p style='color:#CBD5E1;font-size:0.9rem;line-height:1.65;margin:0 0 0.75rem 0;'>
            {benchmark_text}
        </p>
        <p style='color:#CBD5E1;font-size:0.9rem;line-height:1.65;margin:0;'>
            {quote_summary}
            {quote_impact}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

# ── Evidence Section ──────────────────────────────────────────────────────────
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;'>
    <div style='width:3px;height:20px;background:linear-gradient(180deg,#3DDAB4,#06B6D4);
         border-radius:2px;'></div>
    <span style='color:#94A3B8;font-size:0.7rem;font-weight:600;
         letter-spacing:0.12em;text-transform:uppercase;'>Model Evidence</span>
</div>
""", unsafe_allow_html=True)

def chart_card(title, subtitle, path, missing_msg):
    exists = os.path.exists(path)
    st.markdown(f"""
    <div style='padding:1rem 1.25rem 0.75rem;
         background:rgba(30,41,59,0.5);
         border:1px solid rgba(61,218,180,0.12);
         border-radius:14px; margin-bottom:0.25rem;'>
        <div style='font-size:0.78rem;font-weight:600;color:#F1F5F9;
             margin-bottom:2px;'>{title}</div>
        <div style='font-size:0.7rem;color:#475569;'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
    if exists:
        st.image(path, use_column_width=True)
    else:
        st.markdown(f"""
        <div style='padding:2rem;text-align:center;
             background:rgba(30,41,59,0.3);border-radius:10px;
             border:1px dashed rgba(61,218,180,0.15);margin-top:4px;'>
            <div style='color:#3DDAB4;font-size:1.25rem;margin-bottom:0.5rem;'>⧖</div>
            <div style='color:#475569;font-size:0.78rem;'>{missing_msg}</div>
        </div>
        """, unsafe_allow_html=True)

core_col1, core_col2 = st.columns(2, gap="medium")
with core_col1:
    chart_card(
        "Feature Importance — SHAP",
        "Which signals drive default risk most?",
        "visuals/shap_summary.png",
        "Run generate_visuals.py"
    )
with core_col2:
    chart_card(
        "Default Rate Over Time",
        "The 2008 financial crisis in the data",
        "visuals/time_series.png",
        "Run generate_visuals.py"
    )

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;'>
    <div style='width:3px;height:20px;background:linear-gradient(180deg,#06B6D4,#3DDAB4);
         border-radius:2px;'></div>
    <span style='color:#94A3B8;font-size:0.7rem;font-weight:600;
         letter-spacing:0.12em;text-transform:uppercase;'>Geographic & Industry Analysis</span>
</div>
""", unsafe_allow_html=True)

eda_col1, eda_col2 = st.columns(2, gap="medium")
with eda_col1:
    chart_card("Default Rate by Industry", "Which sectors carry the most risk?",
               "visuals/industry_risk.png", "Run generate_eda.py")
with eda_col2:
    chart_card("Default Rate by State", "Geographic bias — the core finding",
               "visuals/state_risk.png", "Run generate_eda.py")

eda_col3, eda_col4 = st.columns(2, gap="medium")
with eda_col3:
    chart_card("Loan Size Distribution", "The range of loans in the dataset (log scale)",
               "visuals/loan_distribution.png", "Run generate_eda.py")
with eda_col4:
    chart_card("Feature Correlation Heatmap", "How engineered features relate to each other",
               "visuals/correlation_heatmap.png", "Run generate_eda.py")

# ── Choropleth ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;'>
    <div style='width:3px;height:20px;background:linear-gradient(180deg,#FBBF24,#06B6D4);
         border-radius:2px;'></div>
    <span style='color:#94A3B8;font-size:0.7rem;font-weight:600;
         letter-spacing:0.12em;text-transform:uppercase;'>Interactive US Default Rate Map</span>
</div>
""", unsafe_allow_html=True)

if os.path.exists('visuals/choropleth.html'):
    with open('visuals/choropleth.html', 'r') as f:
        st.components.v1.html(f.read(), height=500)
else:
    st.markdown("""
    <div style='padding:3rem;text-align:center;
         background:rgba(30,41,59,0.3);border-radius:14px;
         border:1px dashed rgba(61,218,180,0.15);'>
        <div style='color:#3DDAB4;font-size:2rem;margin-bottom:0.75rem;'>🗺</div>
        <div style='color:#475569;font-size:0.85rem;'>
            Interactive choropleth coming soon · run <code style='color:#3DDAB4;'>generate_map.py</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem;padding:1.5rem 0;
     border-top:1px solid rgba(61,218,180,0.1);
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;'>
    <span style='color:#1E3A2F;font-size:0.72rem;font-weight:700;
         letter-spacing:0.1em;text-transform:uppercase;'>FairRate</span>
    <span style='color:#334155;font-size:0.7rem;'>
        AI Community Datathon 2026 · Stony Brook University ·
        SBA Dataset 1987–2014 · 900,000+ loans ·
        XGBoost Classifier + Regressor
    </span>
</div>
""", unsafe_allow_html=True)
