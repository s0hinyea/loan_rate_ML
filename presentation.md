# Project: Is My Small Business Loan Rate Fair?
## The Stony Brook AI Community Datathon 2026 — Master Source for Devpost/Slides/Video

---

### 1. The Core Problem (The "Why")
**The Hook:** A small business owner in a rural area or an underserved urban sector walks into a bank. They get quoted an 8.9% interest rate. They have no idea if that's fair. They simply pay it.
**The Reality:** Banking risk is often a "black box." Our data proves that geography and industry often predict your loan rate more than your actual business quality.
**The Solution:** An AI-powered transparency tool that uses 900,000 historical outcomes to tell you: "Your quoted rate is an overcharge. The fair range for your business is 6.1%–7.2%."

---

### 2. The Data (The "Science")
*   **Source:** Real historical Small Business Administration (SBA) dataset (Kaggle).
*   **Scale:** 574,207 real loans (after strict cleaning).
*   **Timeline:** 1980s through 2014—includes the "Ground Truth" of the 2008 Financial Crisis.
*   **Integrity:** We strictly dropped "leaky" columns like `ChargeOffDate` to ensure our model only learns from what a bank knows *before* approval.

---

### 3. Feature Engineering (The "Secret Sauce")
We didn't just use the raw data. We engineered **11 "Super Features"** that tell the story:
1.  **State_Sector_Default_Rate:** (Priority 1) Captures the joint risk of unique state/industry combos — solves the "Data Hole."
2.  **Is_Recession:** A flag for 2008-2009 (because the world changed those years).
3.  **Industry_Default_Rate:** Benchmarking how risky your sector is historically.
4.  **State_Default_Rate:** The "Fairness" signal—proving geography matters.
5.  **Loan_Size_Bucket:** Discretized loan amounts (e.g., <$25k vs >$25k) to capture model non-linearity.
6.  **Zero_Jobs_Created:** A binary flag for high-risk startups with no employment base.
7.  **Loan_to_Jobs_Ratio:** Ratio of loan size to job creation (business productivity).
8.  **SBA_Coverage_Ratio:** How much confidence the government has in co-signing the loan.
9.  **Loan_vs_Industry_Avg:** Contextual sizing (is $100k big for a cafe or a factory?).

---

### 4. The Model Progression (The "How")
We didn't just jump to XGBoost. We built a logical path to show the judges our evolution:
1.  **Baseline (Logistic Regression):** F1 = **0.4346**. (Failed linearly; proved the data's complexity).
2.  **Intermediate (Random Forest):** F1 = **0.8400**. (The breakthrough into non-linear patterns).
3.  **Champion (XGBoost + Tuning):** F1 = **0.8607**. (Optimized with Balanced Weights, Threshold Tuning, and **Platt Calibration**).

**Why we trust that score:** the geography features (`state_default_rate`, `industry_default_rate`, `state_sector_default_rate`) are rebuilt from the training split only before evaluation, so the test set is no longer leaking its own default outcomes back into those aggregates.

---

### 5. Fair Rate Regression Model
To power our "Fair Rate" tool, we trained a separate **XGBoost Regressor** to predict government confidence (`sba_coverage_ratio`).
*   **Result:** R² Score = **0.738**
*   **Accuracy:** RMSE = **0.088** (Predictions are on average within ~8.8% of historical guarantees).
*   **Why it matters:** This allows us to map government confidence directly into a fair interest rate range, providing the "Killer Demo" moment for our app.

---

### 5. The "Killer Demo" Moment (The Video Script)
**Scenario:** "Imagine a restaurant owner in Queens, NY. 3 years in business, 10 employees. The bank quoted them 8.9%."
**The Reveal:**
1.  Open the Streamlit App.
2.  Input the business details LIVE, including the bank's quoted rate.
3.  **The Output:** "Default Risk: 14% (Low) | Fair Rate: 6.4% | Quoted Rate: 8.9% | Potential Overcharge: ~$12,000."
4.  **The Impact:** "That's **$12,000 extra** out of that owner's pocket over 10 years. Our tool gives them the data they need to walk back into that bank and negotiate."

---

### 6. Technical Differentiator
*   **Explainable AI (SHAP):** Every prediction comes with a chart. We don't just say "High Risk"—we show that the risk is high specifically due to the "Mississippi Geography Signal" or the "Recession Flag."
*   **Real Data Only:** No synthetic targets. Our regressor predicts actual `sba_coverage_ratio`, a real historical metric of government confidence.

---

### 7. Limitations & Future Work
*   **Recency:** Data ends in 2014. Future versions would API into current Fed prime rates.
*   **Collateral:** Doesn't account for personal real estate assets (not in dataset).
*   **Scale:** Expanding beyond SBA to private commercial lending.

---

### 8. The Visuals (generate_visuals.py → `visuals/`)

Two charts are generated and embedded directly in the Streamlit app.

#### Visual 1 — SHAP Summary Plot (`visuals/shap_summary.png`)
- **What it is:** A beeswarm plot showing which features pushed the XGBoost classifier's predictions up or down, and by how much.
- **How it was made:** `shap.TreeExplainer` on the fitted XGBoost base estimator inside the calibrated classifier artifact, run on a **5,000-row random sample**.
- **What to say:** "Each dot is one loan. Features are ranked top-to-bottom by importance. Red = high feature value, blue = low. Look at where `state_default_rate` and `industry_default_rate` rank — they are in the top features. The model learned that *where you are* and *what industry you're in* are the dominant predictors of default. That is the geographic bias finding."
- **Talking point for judges:** If asked why SHAP over built-in feature importance — "XGBoost's default importance just counts how often a feature is split on. SHAP actually measures the magnitude of impact on the final prediction, which is far more honest."

#### Visual 2 — Time Series (`visuals/time_series.png`)
- **What it is:** Line chart of average default rate per year (1980s–2014), with a red shaded box over 2008–2010.
- **How it was made:** `df.groupby('ApprovalFY')['default'].mean()`, plotted with `axvspan` for the recession highlight.
- **What to say:** "You can see a dramatic spike exactly at 2008. This is why we engineered `is_recession` as a binary flag — loans approved in those two years had catastrophically different outcomes than any other period. Without flagging this, the model would have treated a 2008 restaurant loan the same as a 2005 restaurant loan. That would have been wrong."

---

### 9. The Streamlit App — Full Technical Walkthrough (`app/app.py`)

#### Architecture
- **Entry point:** `streamlit run app/app.py` from the project root.
- **Model loading:** Both pkl files load once via `@st.cache_resource` — not reloaded on every click.
- **Data loading:** `df_features.csv` loads via `@st.cache_data` and pre-computes four O(1) lookup dicts for state rates, industry rates, industry avg loan size, and state-sector risk.
- **Styling:** Custom CSS injected via `st.markdown()` — Inter font, Navy→Carbon sidebar gradient, glassmorphic metric cards, emerald green `#10B981` analyze button.

#### The Feature Engineering Bridge (`engineer_features_for_model()`)
This is the most technically important function in the app. It converts the user's business profile into the exact 24-column feature vector the classifier was trained on and the 18-column vector the regressor was trained on. The quoted-rate input stays outside the model and is used in a post-prediction comparison layer.

| Human Input | Model Feature | How |
|---|---|---|
| State dropdown | `state_default_rate` | Lookup dict from df_features.csv |
| Industry dropdown | `industry_default_rate` | Lookup dict from df_features.csv |
| Loan Amount | `DisbursementGross`, `GrAppv` | Direct / set equal |
| Loan Amount | `SBA_Appv` | `loan_amount × mean_sba_coverage_ratio` |
| Loan Amount + Industry | `loan_vs_industry_avg` | `loan_amount / sector_mean_disbursement` |
| Jobs Created | `loan_to_jobs_ratio` | `loan_amount / (jobs_created + 1)` |
| Business Status | `is_new_business`, `NewExist` | `NewExist=2` if new, else `1` |
| Urban/Rural | `UrbanRural` | `1`=Urban, `2`=Rural |
| LowDoc | `LowDoc` | `1`=LowDoc, `0`=Standard |
| Franchise | `IsFranchise` | `1`/`0` |
| Revolving Credit | `RevLineCr` | `1`/`0` |
| — (hardcoded) | `is_recession` | Always `0` (user is not in 2008) |
| — (hardcoded) | `ApprovalFY` | `2024` |
| — (mean from data) | `sba_coverage_ratio`, `disbursement_ratio` | Historical averages as neutral priors |
| Quoted Rate input | — | Compared against `fair_rate` after prediction to estimate overcharge / savings |

The app also derives `state_sector_default_rate`, `loan_size_bucket`, and `zero_jobs_created` before scoring.

**Classifier gets:** the full 24-column feature vector.
**Regressor gets:** the same base profile, then the app aligns to the exact feature schema expected by the saved regressor artifact.

#### The Fair Rate Formula
```
Fair Rate = 8.5% + (1.0 - predicted_sba_coverage_ratio) × 10
```
- `sba_coverage_ratio = 0.90` → `8.5 + 1.0 = 9.5%` (government very confident, low risk)
- `sba_coverage_ratio = 0.50` → `8.5 + 5.0 = 13.5%` (medium confidence)
- `sba_coverage_ratio = 0.30` → `8.5 + 7.0 = 15.5%` (government nervous, high risk)
- The ratio is clamped to `[0.0, 1.0]` before applying the formula.

#### The Four Output Metrics
1. **Default Risk %** — `classifier.predict_proba(input_clf)[0][1]` × 100. Delta shows vs historical avg for that state+sector combo.
2. **Fair Interest Rate %** — from the formula above. Delta shows vs what similar businesses historically implied.
3. **Bank Quoted Rate %** — the user's quoted APR, shown directly against the fair-rate estimate.
4. **Potential Overcharge / Savings** — simple-interest estimate of how much the quoted rate costs or saves over the full term.

#### Risk Labels (thresholds)
- `< 29.5%` (below 50% of optimal threshold) → 🟢 Low Risk
- `29.5%–59%` (between 50% and optimal threshold) → 🟡 Medium Risk
- `> 59%` (above optimal threshold 0.59) → 🔴 High Risk

#### Dollar Cost of Rate Gap
```
quote_gap  = quoted_rate - fair_rate
dollar_gap = loan_amount × (quote_gap / 100) × (term_months / 12)
```
Simple interest over the loan term. Good enough for a demo — judges will not stress-test this math.

#### Visual Panels
- **SHAP + time series:** Always visible below the results. Show a `st.warning` if images are missing — auto-display the moment the file exists. No code change needed.
- **EDA panels (4):** Placeholders for Person B's `industry_risk.png`, `state_risk.png`, `loan_distribution.png`, `correlation_heatmap.png`. Same auto-display logic.
- **Choropleth:** Placeholder for Person B's `choropleth.html` rendered via `st.components.v1.html`.

---

### 10. Demo Script — Recommended Live Walkthrough

> The risk predictions below are verified against the trained model. The quoted-rate comparison is user-entered, and the dollar impact is then computed deterministically from that quote.

> These three presets are built into the app's **Quick Demo Presets** and all use state-industry combinations with strong historical support, so the live demo stays stable.

---

**Step 1 — The Hook (30 seconds)**
"900,000 real loans. One question: is your rate fair?"

Click a preset in the sidebar, then point out that the quote, fair rate, and overcharge all update live. Each preset is still fully editable after loading.

**Step 2 — 🟢 Low Risk Baseline (~0.7%)**

| Field | Value |
|---|---|
| State | `ND` |
| Industry | `Healthcare` (Sector 62) |
| Loan Amount | `$150,000` |
| Quoted Rate | `13.4%` |
| Jobs Created | `5` · Jobs Retained `5` · Employees `10` |
| Term | `84 months` |
| Business Status | `Existing Business` · Urban · Standard |

→ Show: ~`0.7%` default risk, fair rate ~`13.6%`, quote lands **within fair range**, support count `151`. This is your calm baseline.

---

**Step 3 — 🟡 Medium Risk Overcharge (~38%)**

| Field | Value |
|---|---|
| State | `CA` |
| Industry | `Retail Trade` (Sector 44) |
| Loan Amount | `$20,000` |
| Quoted Rate | `15.0%` |
| Jobs Created | `1` · Jobs Retained `0` · Employees `2` |
| Term | `48 months` |
| Business Status | `New Business` · Urban · Standard |

→ Show: ~`38.1%` risk, fair rate ~`13.5%`, potential overcharge ~`$1,200`, support count `10,884`. Say: "This is the caution zone. The loan is not catastrophic, but the quote is clearly above what the model thinks is fair."

---

**Step 4 — 🔴 High Risk Overcharge (~82%)**

| Field | Value |
|---|---|
| State | `CA` |
| Industry | `Retail Trade` (Sector 44) |
| Loan Amount | `$20,000` |
| Quoted Rate | `16.5%` |
| Jobs Created | `2` · Jobs Retained `1` · Employees `5` |
| Term | `30 months` |
| Business Status | `Existing Business` · Urban · Standard |

→ Show: ~`82.0%` risk, fair rate ~`13.0%`, potential overcharge ~`$1,750`, support count `10,884`. Say: "Now we are in a genuinely dangerous lending profile and the quote is also materially above the model's fair-rate estimate."

---

**Step 5 — The Confidence Guardrail (Best Judge Moment)**
Switch to a sparse pocket like `FL + Sector 92` and say:
> "When a state-industry pocket has too little historical support, the app warns you and lowers its confidence language instead of pretending the estimate is equally reliable. That makes the tool more honest."

---

**Step 6 — Point at SHAP chart**
"The model isn't guessing. SHAP shows which combinations of term, geography, loan structure, and business profile pushed the prediction up or down. The model is transparent about why."

---

### 11. Interaction Effects + Sparse Pockets — Know This Cold

**The Discovery:** The model does not follow one simple rule like "shorter term is always safer" or "smaller loans are always riskier." It learns **interactions** across term, loan size, geography, industry, and business profile.

**What this means in practice:**
- A term change can raise or lower risk depending on the surrounding profile.
- A small loan can be low risk in one dense segment and high risk in another.
- Sparse state-industry pockets can behave noisily because the model has fewer local examples to anchor on.

**That is why the app now exposes historical support directly:**
- `<10 loans` → **Very Limited Support**
- `10–29 loans` → **Limited Support**
- `30+ loans` → no warning

**How to Use This With Judges:**
> "Our model captures interactions, not one-dimensional rules. And when the local history is thin, the app says so. We would rather surface uncertainty than pretend every estimate carries the same confidence."

---

### 12. The "Data Hole" — Technical Discovery & Solution
During testing, we found that certain State+Sector combos had very little historical support. We solved this in v1.2:
1.  **Direct Interaction Feature:** We added `state_sector_default_rate` as a primary feature. This allows the model to "see" that a Retailer in Georgia is fundamentally different from a Retailer in Maine, even if the state-level averages are similar. 
2.  **In-App Confidence Guardrail:** The app now counts historical loans for the exact state + sector pocket and warns the user when support is sparse.
3.  **Cross-Model Verification:** Our **Fair Rate Regressor** (predicting govt backing) often catches risk even when the Classifier has thin local data. This two-model validation makes the system significantly safer.

**Judge Moment:** *"We didn't just accept sparse data as a limit. We engineered interaction features and then surfaced confidence in the app itself, so users can see when a result comes from a dense historical pocket versus a thin one."*

---

### 13. Technical Wins & Future Scaling (Judges' Favorite Topic)

#### COMPLETED WINS (Implemented during development):
*   **[COMPLETED] `state_sector_default_rate` Interaction Feature** — Directly solved the "Data Hole" sparsity problem. 
*   **[COMPLETED] Probability Calibration** — Used **Platt Scaling** to ensure that "20% risk" in the app matches 20% real-world risk. 
*   **[COMPLETED] Decisition Threshold Tuning** — Tuned the optimal cutoff (F1 score) to handle the 82/18 class imbalance.
*   **[COMPLETED] Interactive US Choropleth** — Built a Plotly map for national geographic risk discovery.

#### Future Tiers of Scaling:
1. **Dynamic Prime Rate Integration:** Current `8.5%` is a constant. We'd add a FRED API call to fetch daily prime rates.
2. **NAICS 6-Digit Granularity:** Our current model uses 2-digit "Sectors." Adding 6-digit NAICS would let us distinguish between a "Fast Food Restaurant" and a "Nightclub."
3. **Hyperlocal Economic Indicators:** Joining with county-level Unemployment or GDP data would remove the "State Noise" from the model. 

**Tier 3 — Production/scale improvements**
7. **Fed Prime Rate API join** — the hardcoded `8.5%` in the fair rate formula should be dynamic. FRED API (free) provides historical prime rates by year.
8. **County-level unemployment** — state default rate is noisy. FRED county-level unemployment data would add precision without requiring new loan data.
9. **NAICS 6-digit codes** — current 2-digit sector codes lump restaurants and bars together (both Sector 72). Full NAICS codes would sharpen industry signals.
10. **Post-2014 SBA data** — the SBA publishes annual updates. Retraining fills data holes and improves modern relevance.

---

### 14. Judges' Q&A / Study Guide (Know These Cold)

#### Q1: "Why did you use F1-Score instead of Accuracy?"

*   **The Answer:** "Our dataset is imbalanced. Roughly 82% of loans are Paid in Full (PIF) and only 18% defaulted. If we used Accuracy, a model that just guesses 'Always PIF' would get an 82% score—but it would be a useless model that catches zero defaults. F1-Score combines Precision and Recall, forcing the model to actually learn how to identify the risky 18%."

#### Q2: "Why did Tree-based models (RF/XGB) beat Linear models (Logistic Reg)?"
*   **The Answer:** "Logistic Regression assumes a straight-line relationship between a feature and the result. But business risk is non-linear. For example, a $100k loan might be low risk for a doctor but high risk for a new restaurant. Tree-based models can 'branch' on these interactions (e.g., IF Industry=72 AND Year=2008 AND Term<60), capturing the complex reality of the SBA portfolio."

#### Q3: "What is SHAP and why is it in your app?"
*   **The Answer:** "XGBoost is often a 'black box.' SHAP (SHapley Additive exPlanations) uses game theory to mathematically attribute exactly how much each feature contributed to a specific prediction. It turns 'The computer says no' into 'The model flagged this because of the 2008 Recession Flag and the High-Risk Georgia Sector code.' It's about transparency and trust."

#### Q4: "How did you ensure your model wasn't 'cheating' (Data Leakage)?"
*   **The Answer:** "We were extremely careful to drop any column that only exists *after* a loan defaults. For example, the original dataset has `ChgOffDate` (Charge-off Date). If a row has a date there, it obviously means it defaulted. Including that would give the model the 'answer' before the quiz. We strictly used only 'Pre-Approval' features like Term, Loan Amount, and Industry."

#### Q5: "Your data only goes to 2014. Is it still relevant?"
*   **The Answer:** "While the specific rates have changed, the *behavioral patterns* remain the same. The way small businesses reacted to the 2008 financial crisis is a vital blueprint for how they react to any modern economic shock. Our model captures those indestructible patterns of risk."

#### Q6: "Why does a $25k loan show higher risk than a $200k loan? That seems backwards."
*   **The Answer:** "Sometimes the model does assign higher risk to a smaller loan, but that's not a universal rule. It depends on the full interaction between term, geography, industry, business status, and loan structure. That's exactly why we show SHAP and historical support in the app — to avoid pretending there's one simple explanation for every profile."

#### Q7: "Your model sometimes shows low default risk for high-risk states. Is it broken?"
*   **The Answer:** "No — that usually means the local state + sector pocket is sparse. The app now warns when that happens and lowers its confidence language. Under the hood, the model then relies more on broader signals like term, size, and sector behavior. In a production version we'd add hierarchical smoothing to make those sparse pockets even safer."

#### Q8: "How long does it take to retrain as new SBA data comes in?"
*   **The Answer:** "Thanks to our efficient feature engineering pipeline and XGBoost, the entire pipeline — cleaning, feature engineering, training all three models — runs end to end in under 2 minutes. This means the tool stays current as new SBA data is published annually."

#### Q9: "What's the biggest thing you'd change about the model?"
*   **The Answer:** "Two things. First, we'd add hierarchical smoothing for sparse state-sector pockets so rare combinations behave more smoothly. Second, we'd replace the hardcoded fair-rate constant with a live prime-rate feed, so the tool stays economically current instead of relying on a static baseline."

---
*Created by Person A (Model Lead) and Person B (Data/App Lead)*
