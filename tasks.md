Person A — You (The Model Lead)
Your domain: Regression, Feature Engineering, SHAP
You own everything that requires deeper model knowledge. Your job is to make the models smart, not just working.

Your Files
analysis.ipynb — the middle and bottom half (feature engineering through model evaluation)
shap_analysis.py — standalone SHAP work you plug into the app later

Your Tasks in Order
Saturday Night (6–10 PM)
Download and open the dataset with your partner
Agree on column names, target variable, and what the cleaned dataframe should look like
Sketch out all the features you plan to engineer so your partner knows what columns to expect
Sunday Morning (10 AM–2 PM)
Take the cleaned dataframe your partner hands you
Build every engineered feature:
python
df['loan_to_jobs_ratio']     = df['DisbursementGross'] / (df['CreateJob'] + 1)
df['is_recession']           = df['ApprovalFY'].isin([2008, 2009, 2020]).astype(int)
df['industry_default_rate']  = df.groupby('Sector')['default'].transform('mean')
df['is_new_business']        = (df['NewExist'] == 2).astype(int)
df['sba_coverage_ratio']     = df['SBA_Appv'] / df['GrAppv']
Run the full model comparison (Logistic → Random Forest → XGBoost) on the classification target
Build the regression model separately predicting rate fairness
Generate the model comparison table
Run SHAP on the winning XGBoost model
Sunday Afternoon (2–5 PM)
Build the time series chart (your advanced feature)
Save both trained models with pickle
Hand model files to your partner to plug into the app
Help with the presentation slides — you own the model results section

Your Deliverables
Trained XGBoost classifier → models/xgb_classifier.pkl
Trained XGBoost regressor → models/xgb_regressor.pkl
Model comparison table (5 rows, clean)
SHAP summary plot image
Time series chart image

What You Should Know Cold
How to interpret F1 score, precision, recall
Why XGBoost beats Random Forest on this data
What SHAP values mean so you can explain it to a judge in 30 seconds
The regression target — what you're actually predicting and why it represents rate fairness

Person B — Your Partner (The Data and App Lead)
Your domain: Classification output, Data Cleaning, EDA, Streamlit App
You own everything from raw data to clean data, all the visualizations, and the entire user-facing tool. You are the reason the project looks impressive and is actually usable.

Your Files
analysis.ipynb — the top half (loading through EDA)
app.py — the entire Streamlit application
requirements.txt

Your Tasks in Order
Saturday Night (6–10 PM)
Download and open the dataset together
Note down every messy thing you find (null counts, weird MIS_Status values, dollar sign columns)
Set up the Streamlit skeleton so it's runnable before Sunday even starts
Sunday Morning (10 AM–2 PM)
Clean the full dataset:
python
# 1. Keep only rows with a real outcome
df = df[df['MIS_Status'].isin(['CHGOFF', 'P I F'])]

# 2. Create binary target
df['default'] = (df['MIS_Status'] == 'CHGOFF').astype(int)

# 3. Strip dollar signs and convert to float
dollar_cols = ['DisbursementGross', 'GrAppv', 'SBA_Appv', 'BalanceGross']
for col in dollar_cols:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

# 4. Extract 2-digit sector from NAICS
df['Sector'] = df['NAICS'].astype(str).str[:2]

# 5. Drop columns that leak the answer
df.drop(columns=['ChgOffDate', 'ChgOffPrinGr', 'BalanceGross'], inplace=True)
Build every EDA chart:
Default rate by industry (bar chart)
Default rate by state (bar chart — choropleth comes later)
Loan size distribution (histogram)
Correlation heatmap
Build the choropleth map (your advanced feature)
Hand the cleaned dataframe to your partner
Sunday Afternoon (2–5 PM)
Receive the trained model pkl files from your partner
Plug them into the Streamlit app
Wire up every input widget to the prediction pipeline
Add the SHAP waterfall plot your partner generates into the app output
Test the full demo flow end to end — make sure the demo moment works perfectly
Sunday Evening (5–9 PM)
Own the presentation slides visually — you have all the charts
Practice the live demo so you can run it smoothly in front of judges
Prepare the "why this finding matters" talking points

Your Deliverables
Fully cleaned df_clean.csv handed to your partner by noon
4 EDA charts saved as images
Choropleth map (interactive, in the notebook and in the app)
Complete working Streamlit app
Smooth rehearsed demo

What You Should Know Cold
Every column in the dataset and what it means
Why you dropped ChgOffDate and ChgOffPrinGr (they leak the answer — only exist after a default)
The demo flow word for word — what you type in, what comes out, what you say
How to explain the choropleth map finding to a non-technical judge

The Handoff Points — These Are Critical
These are the three moments where your work connects. Miss these and you're both blocked.
When
What Gets Handed Off
From → To
Saturday 10 PM
Agreed column names + cleaned data contract
Both → agreed together
Sunday 12 PM
df_clean.csv — fully cleaned, ready for modeling
B → A
Sunday 3 PM
xgb_classifier.pkl + xgb_regressor.pkl
A → B


The One Rule
If you're ever both looking at the same file at the same time you're wasting half your team. The notebook splits cleanly at feature engineering — everything above that line is Person B, everything below is Person A. The app is entirely Person B. Never cross those lines unless you're stuck.

