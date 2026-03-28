# Is My Small Business Loan Rate Fair?

A data-driven transparency tool for small business owners to evaluate the fairness of their loan interest rates using the SBA National Loan Dataset. This project predicts both default risk (classification) and a fair interest rate range (regression) based on historical outcomes, ensuring borrowers understand if their quoted rate is reasonable given their industry, location, and business size.

## Project Structure
- `data/`: Place the SBA National Loan Dataset here.
- `notebooks/`: Contains `analysis.ipynb` for data cleaning, EDA, feature engineering, and model training.
- `models/`: Models trained in the notebook will be saved here as pickle files (`.pkl`).
- `app/`: Contains `app.py`, the interactive Streamlit application for end-user predictions.

## How to Install Dependencies
It is recommended to use a virtual environment. Install the pinned dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run the Notebook
To open and run the main analysis notebook, launch Jupyter Notebook from the project root:
```bash
jupyter notebook notebooks/analysis.ipynb
```
Follow the notebook cells in order to process raw data and save the model.

## How to Launch the Streamlit App
To run the interactive demonstration application (even before the model is completely trained, it uses a placeholder):
```bash
streamlit run app/app.py
```
This will open the app in your default web browser.

## Dataset
Download the SBA (Small Business Administration) National Loan Dataset from Kaggle.
Extract the CSV file into the `data/` directory. The notebook expects this file for its first steps.
