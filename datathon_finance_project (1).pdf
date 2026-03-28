STONY BROOK UNIVERSITY
AI Community Datathon 2026 | Finance Track
  Is My Small Business
  Loan Rate Fair?
  A data-driven transparency tool for small business owners
■ Finance Track ■ Predictive ML ■ Streamlit App ■ SBA Dataset
 01 THE PROBLEM
Why this matters and who it hurts
Every year, millions of small business owners walk into a bank and get quoted an interest rate on a loan.
They have no idea if that rate is fair, too high, or outright predatory. Unlike home mortgages — where
comparison tools are everywhere — small business lending is a black box.
The result: first-generation entrepreneurs, minority-owned businesses, and rural operators are routinely
overcharged compared to equivalent businesses in wealthier or more connected areas. They can't push back
because they have nothing to compare to. They simply accept whatever they're given.
 The Demo Moment
"I'm a restaurant owner, 3 years in business, Queens NY, looking for a $150,000 loan. The bank
quoted me 8.9%. Our model says the fair range is 6.2%–7.4%. That's $12,000 extra out of your
pocket over 10 years."
That specific, personal number is our version of the TTC team's 'you'll be 4 minutes late' moment. It's what makes a judge
remember your project.
900k+
Real SBA Loans
~20%
Historical Default Rate
$12k
Avg Overcharge Found
0.93
Target XGBoost F1
 02 THE DATA
Where it comes from and what's in it
We use the SBA (Small Business Administration) National Loan Dataset, freely available on Kaggle. This
is not a synthetic or educational dataset — these are real approved loans with real outcomes tracked over
decades.
What Each Row Represents
• One approved SBA loan with full details on the business, loan terms, and final outcome
• Outcome column tells us: was the loan fully paid back, or did it default (charged off)?
• Over 900,000 rows spanning the 1980s through 2014 — includes the 2008 financial crisis
Key Columns We Use:
Column What It Means Why It Matters
NAICS / Industry What type of business Some industries default far more than others
State / City Where the business is Geography predicts both risk and rate unfairness
Term Loan length in months Longer terms = more risk exposure
DisbursementGross Total loan amount Larger loans behave differently than small ones
NoEmp Number of employees Proxy for business size and stability
NewExist New vs existing business New businesses default at much higher rates
MIS_Status Paid in full vs charged off This is our TARGET variable
ApprovalFY Year the loan was approved Lets us account for recession years (2008, 2020)
 03 HOW WE BUILD IT
Step by step from raw data to working app
Step 1 — Load & First Look
→ df.shape, df.info(), df.describe(), df.isnull().sum()
→ Understand what every column means before touching anything
→ Note: MIS_Status is our target. 'CHGOFF' = default, 'PIF' = paid in full
Step 2 — Clean the Data
→ Remove rows where MIS_Status is missing (can't train without a label)
→ Convert dollar columns like DisbursementGross from strings ('$150,000') to numbers
→ Flag recession years: 2008, 2009, 2020 get their own binary column
→ Drop columns that leak future info (e.g. ChargeOffDate — only exists if already defaulted)
Step 3 — Explore the Data (EDA)
→ Default rate by industry — which sectors are riskiest?
→ Default rate by state — is geography a real predictor?
→ Loan size distribution — are there outlier mega-loans skewing everything?
→ Correlation heatmap — which numeric features move together?
→ New vs existing businesses — how much does business age matter?
Step 4 — Feature Engineering
→ loan_to_jobs_ratio = DisbursementGross / NoEmp (cost per job created)
→ is_recession = 1 if ApprovalFY in [2008, 2009, 2020]
→ industry_default_rate = average default rate for that industry (group mean)
→ is_new_business = 1 if NewExist == 1
→ loan_size_bucket = small / medium / large based on percentiles
 04 THE MODELS
What we build, in what order, and why
We follow the same model progression that both winning datathon teams used. Start simple, prove it works,
then upgrade. Every step has a reason.
Model Type Purpose Expected Score
Logistic Regression Classification Baseline — will it default? ~0.72 F1
Random Forest Both Robust default + rate prediction ~0.85 F1
XGBoost Both Maximum accuracy — final model ~0.93 F1
What we're predicting (two separate models):
Classification Model Regression Model
Will this loan default? (yes/no) What interest rate should this loan get?
Target: MIS_Status (CHGOFF vs PIF) Target: Interest rate / GrAppv ratio
Metric: F1 score — balances precision and
recall
Metric: R squared score (target > 0.90)
Output in app: default risk percentage Output in app: fair rate range
The key finding we expect to surface:
When we look at feature importance from XGBoost, industry and geography will almost certainly rank higher
than pure financial metrics. That means where your business is located and what sector you're in predicts
your loan rate more than your actual creditworthiness. That finding is the centerpiece of our presentation.
 05 THE STREAMLIT APP
Turning the model into a tool anyone can use
Both winning projects — the TTC transit team and the Toronto real estate team — ended with a live
interactive tool. Not just charts. Not just a notebook. A tool where you put in YOUR numbers and get YOUR
answer. That's what we build.
User Input App Logic Output
Industry type
Years in business
Loan amount
Number of employees
State / location
Load pre-trained XGBoost model
Encode inputs to match training data
Run classification model
Run regression model
Default risk: 23%
Fair rate range: 6.2%–7.4%
Your quoted rate vs fair rate
Dollar difference over loan term
Minimum Streamlit code to get running in under an hour:
import streamlit as st
import pickle, pandas as pd
model = pickle.load(open('xgb_model.pkl', 'rb'))
st.title('Is My Business Loan Rate Fair?')
industry = st.selectbox('Industry', industries)
years = st.slider('Years in Business', 0, 30, 5)
loan_amt = st.number_input('Loan Amount ($)', 10000, 5000000, 150000)
employees = st.number_input('Number of Employees', 1, 500, 10)
state = st.selectbox('State', states)
if st.button('Analyze My Loan'):
risk = model.predict_proba(input_df)[0][1]
st.metric('Default Risk', f'{risk:.1%}')
st.metric('Fair Rate Range', f'{fair_lo:.1f}% - {fair_hi:.1f}%')
 06 FULL TECH STACK
Every tool we use and exactly what it does
Tool / Library What It Does In This Project
Python Core language everything is written in
Pandas Load, clean, and manipulate the SBA dataset
Scikit-learn Logistic Regression, Random Forest, train/test split, metrics
XGBoost Final high-accuracy model for default and rate prediction
Matplotlib / Seaborn EDA charts — correlation heatmap, default rates by industry
Streamlit Interactive web app — the live demo tool
Pickle Save trained model so Streamlit can load it instantly
Jupyter Notebook Development environment — where all analysis is built
SBA National Dataset 900k+ real approved loans with outcomes — primary data source
Note: All of these are free, open source, and installable with pip. No paid APIs, no cloud credits needed.
 07 24-HOUR TIMELINE
Exactly how two people divide and conquer this
Time Who Tasks
SAT 6–10 PM Both Pick question · Download SBA data · First look · Divide roles
SUN 10AM–2PM Person A Clean data · Handle nulls/types · Encode categoricals · EDA charts
SUN 10AM–2PM Person B Research SBA columns · Sketch features · Build Streamlit skeleton
SUN 2PM–6PM Person A Feature engineering · Correlation heatmap · Final model-ready CSV
SUN 2PM–6PM Person B Train Logistic Reg → Random Forest → XGBoost · Pick winner
SUN 6PM–9PM Both Plug model into app · Build demo flow · Slides · Practice demo
The Three Things That Will Kill You If You're Not Careful
• Data cleaning taking 6 hours together instead of splitting up — divide immediately
• Scope creep: more features, prettier app, extra models — resist everything after hour 16
• Not having the demo moment scripted — practice the exact words before you present
Minimum viable version that still wins:
Must Have Nice to Have Cut If Short on Time
Clean SBA dataset
3 EDA charts
Random Forest model
Streamlit app + demo
XGBoost over Random Forest
Geographic map visualization
Rate fairness model
LightGBM / Neural net
Extra supplementary datasets
App visual polish
 08 THE PRESENTATION STORY
How to structure the 5 minutes you have in front of judges
1
Open with the pain point (30 sec)
Don't start with methodology. Start with: 'A small business owner walks into a bank. They get
quoted a rate. They have no idea if it's fair.' Make the judge feel the problem.
2
Show the data briefly (45 sec)
900,000 real SBA loans. Decades of outcomes. Not a toy dataset — real money, real businesses,
real defaults. One slide, three bullet points.
3
Show your best EDA finding (60 sec)
The industry vs geography default rate chart. 'Restaurants in Mississippi default at 3x the rate of
consulting firms in New York — but is that risk or is that discrimination?' That question is what your
model answers.
4
Show the model progression (45 sec)
Logistic Regression gave us X. Random Forest gave us Y. XGBoost gave us 0.93. Each step had
a reason. Show the feature importance chart — geography ranked #2.
5
Live demo — the killer moment (90 sec)
Pull up the Streamlit app. Ask a judge or audience member for their state and a business type.
Plug in numbers live. Show the output. This is your TTC moment. Make it personal.
6
Limitations + what's next (30 sec)
Data only goes to 2014. No real-time rates. Doesn't account for collateral. Saying this makes you
sound like a real data scientist, not a student who got lucky.
 09 MORE FEATURES
Three additions that push this project above the rest
These three additions are what separate a good datathon project from a memorable one. None of the other
teams in these examples had all three. Each one takes under two hours to implement but dramatically raises
the technical and visual impression of the project.
 1 — Time Series: We Understood Our Data Historically
What it is in simple terms:
A line graph showing how loan default rates changed year by year from the 1980s through 2014. Most teams
treat every row of data as equal regardless of when it happened. We don't. We show that time matters —
especially around the 2008 financial crisis.
Why we're doing it:
When that 2008 spike appears visibly in our chart, it becomes a presentation moment. We can point to it and
say: 'You can literally see the financial crisis right here in our data.' No other team will have this. It proves our
dataset is real, adds historical depth, and directly justifies why we created a recession feature in our model.
The code:
yearly_defaults = df.groupby('ApprovalFY')['default'].mean()
fig, ax = plt.subplots(figsize=(10, 4))
yearly_defaults.plot(ax=ax, color='steelblue', linewidth=2)
ax.axvspan(2008, 2010, alpha=0.2, color='red', label='2008 Financial Crisis')
ax.set_title('Default Rate Over Time')
ax.set_ylabel('Default Rate')
ax.legend()
■
What it looks like in the presentation
A line graph with a red shaded zone over 2008–2010. Clean, simple, and the spike tells the
story by itself. No explanation needed — the chart does the talking.
 2 — Choropleth Map: We Proved Geography Matters Visually
What it is in simple terms:
A map of the United States where each state is colored based on its loan default rate. Darker red means
more defaults. Lighter means fewer. Like the election night maps you've seen on TV — except ours shows
where small businesses are failing.
Why we're doing it:
Our entire fairness argument depends on proving that geography predicts loan outcomes. We could say that
in words or show a bar chart — or we could put it on a map of the entire country where the pattern is
impossible to miss in two seconds. It's also interactive — judges can hover over any state and see the exact
numbers. The real estate team added TTC subway data as their 'extra initiative' move. This is ours.
The code:
import plotly.express as px
state_stats = df.groupby('State').agg(
default_rate=('default', 'mean'),
loan_count=('default', 'count')
).reset_index()
fig = px.choropleth(
state_stats,
locations='State',
locationmode='USA-states',
color='default_rate',
scope='usa',
color_continuous_scale='Reds',
title='Default Rate by State'
)
fig.show()
■
■
What it looks like in the presentation
You pull it up and say: "Here is default rate by state." Mississippi and certain Southern states
light up dark red. New England barely shows color. The map makes your argument before you
say a word.
 3 — SHAP: We Can Explain Every Single Prediction
What it is in simple terms:
Most models are a black box — they give you an answer but never explain how they got there. SHAP opens
that black box. For every single prediction, it tells you exactly which factors pushed the result up and which
ones pushed it down, and by exactly how much. It answers 'why did the model say that?' for every individual
case.
Why we're doing it:
Neither winning project had this. The real estate team could say 'bedrooms matter in general' but couldn't tell
you why this specific house was priced the way it was. We will be able to say: 'This loan was flagged as high
risk mainly because of the state and industry — not the loan size or number of employees.' In the Streamlit
app, every prediction comes with a chart explaining the reasoning. That turns a number into an explanation.
That is our biggest technical differentiator.
The code:
pip install shap
import shap
# Explain the whole model
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
# Summary chart — shows which features matter most overall
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
# In Streamlit — explain ONE specific prediction
st.subheader('Why this prediction?')
shap.waterfall_plot(explainer(input_df.iloc[0]))
st.pyplot()
■
What it looks like in the presentation
A colorful bar chart where green bars say "this pushed the risk DOWN" and red bars say "this
pushed the risk UP." Simple to read, looks incredibly sophisticated. No other team in the room
will have this.
Time budget for all three additions:
Feature Who When Time
Choropleth Map Person A Sunday 12:00–12:30 PM 30 mins
Time Series Chart Person A Sunday 12:30–1:15 PM 45 mins
SHAP Setup + Summary Plot Person B Sunday 1:00–2:30 PM 90 mins
SHAP in Streamlit App Person B Sunday 2:30–3:30 PM 60 mins
All three fit inside Sunday morning before you touch the presentation. Total cost: about 3.5 hours. Total impact: the difference
between a good project and a winning one.
 What Wins Datathons
✓ A tool, not just a notebook ✓ A specific, personal demo moment ✓ A finding that surprises people
✓ Model progression with a reason ✓ Honest about what doesn't work ✓ Real data, not toy data
AI Community Datathon 2026 · Stony Brook University · Finance Track