import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ADVANCED FEATURE: Import SHAP when ready
# import shap
# import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Loading the Model and Dependencies
# ----------------------------------------------------
# TODO: Load pre-trained XGBoost model from '../models/xgb_model.pkl'. 
try:
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    
# TODO: Load any saved encoders or lists for dropdowns (e.g., states, industries)

# Mocked lists for the skeleton UI to run immediately
industries = ['Consulting', 'Restaurant', 'Retail', 'Manufacturing', 'Technology']
states = ['NY', 'CA', 'TX', 'FL', 'MS']

# ----------------------------------------------------
# 2. Application UI Setup
# ----------------------------------------------------
st.title('Is My Business Loan Rate Fair?')
st.write("A data-driven transparency tool to see if your quoted small business loan rate matches historical fair rates.")

# ----------------------------------------------------
# 3. User Inputs
# ----------------------------------------------------
st.sidebar.header("Enter Business Details")

# TODO: Industry type dropdown
industry = st.sidebar.selectbox('Industry', industries)

# TODO: Years in business slider
years = st.sidebar.slider('Years in Business', 0, 30, 5)

# TODO: Loan amount number input
loan_amt = st.sidebar.number_input('Loan Amount ($)', min_value=10000, max_value=5000000, value=150000)

# TODO: Number of employees number input
employees = st.sidebar.number_input('Number of Employees', min_value=1, max_value=500, value=10)

# TODO: State / location dropdown
state = st.sidebar.selectbox('State', states)

# TODO: User's quoted rate for comparison
quoted_rate = st.sidebar.number_input('Quoted Interest Rate (%)', min_value=0.0, max_value=30.0, value=8.9)

# ----------------------------------------------------
# 4. Predict Button
# ----------------------------------------------------
if st.sidebar.button('Analyze My Loan'):
    st.header("Loan Analysis Results")
    
    # ----------------------------------------------------
    # 5. Running the Models
    # ----------------------------------------------------
    # TODO: Encode inputs to match the training data format and create an input dataframe
    # input_df = pd.DataFrame([{...}])
    
    if model_loaded:
        # TODO: Run the classification model for default risk
        # risk = model.predict_proba(input_df)[0][1]
        
        # TODO: Run the regression model for fair rate
        # fair_rate = rate_model.predict(input_df)[0]
        
        risk = 0.23        # Replace with actual model output
        fair_lo = 6.2      # Replace with actual model output
        fair_hi = 7.4      # Replace with actual model output
    else:
        # PLACEHOLDER PREDICTION (to allow UI testing before model is trained)
        st.info("Using placeholder predictions. Train and save the XGBoost model to use real values.")
        risk = 0.23
        fair_lo = 6.2
        fair_hi = 7.4
        
    # ----------------------------------------------------
    # 6. Outputs Section
    # ----------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        # TODO: Display default risk percentage
        st.metric('Default Risk', f'{risk:.1%}')
        
    with col2:
        # TODO: Display fair rate range
        st.metric('Fair Rate Range', f'{fair_lo:.1f}% - {fair_hi:.1f}%')
        
    # TODO: Show comparison between their quoted rate and fair rate
    st.subheader("Cost of Unfairness")
    if quoted_rate > fair_hi:
        overcharge_pct = quoted_rate - fair_hi
        # TODO: Calculate approximate dollar difference over the loan term
        # (Loan Amount * Overcharge % * 10 years / 100)
        dollar_diff = loan_amt * (overcharge_pct / 100) * 10 
        
        st.error(f"You are being overcharged! Your quoted rate of {quoted_rate:.1f}% is higher than the fair maximum.")
        st.metric("Estimated Extra Cost (10 yrs)", f"${dollar_diff:,.2f}")
    else:
        st.success("Your quoted rate appears fair based on historical data.")

    # ----------------------------------------------------
    # 7. SHAP Explanations
    # ----------------------------------------------------
    # ADVANCED FEATURE
    st.subheader('Why this prediction?')
    st.write("Understand the factors pulling your risk up or down.")
    
    # TODO: Create SHAP waterfall plot for this individual prediction
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer(input_df.iloc[0])
    # shap.waterfall_plot(shap_values)
    # st.pyplot()
