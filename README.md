# FairRate: AI-Powered Small Business Loan Fairness 🏛️📊🚀

**Live Application:** [https://loanrateml.streamlit.app/](https://loanrateml.streamlit.app/)

A data-driven transparency tool for small business owners to evaluate the fairness of their loan interest rates using the SBA National Loan Dataset (574,207 verified outcomes). 

FairRate uses an **XGBoost Classifier (0.86 F1)** to predict default risk and an **XGBoost Regressor (0.738 R²)** to calculate a fair interest rate range based on historical government confidence.

## Project Structure
- `app/`: The interactive Streamlit application.
- `notebooks/analysis.ipynb`: Full analytical pipeline (Cleaning, EDA, Training, Calibration).
- `models/`: Production-ready XGBoost models (`.pkl`).
- `data/df_clean.csv`: The processed, leakage-free dataset.
- `visuals/`: Core EDA and SHAP explainability charts.

## Key Technical Wins
- **0.86 F1 Score:** Optimized for imbalanced lending data using threshold tuning.
- **Platt Calibration:** Ensures predicted risk % has real-world probabilistic meaning.
- **Leakage-Free:** Strict removal of post-outcome columns to ensure valid predictive power.
- **Glassmorphic UI:** A premium, modern FinTech user experience.

## Quick Start
1.  **Install:** `pip install -r requirements.txt`
2.  **Run App:** `streamlit run app/app.py`
3.  **Explore Data:** Open `notebooks/analysis.ipynb`.

---
**SBA Dataset 1987–2014 · AI Community Datathon 2026**
