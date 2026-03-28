Team Reference Document
Saturday Night Agreement — Read This Together Before Splitting Off

1. Columns We Keep
1.1 Columns That Survive Cleaning
ColumnWhat It Actually IsStateWhere the business is locatedNAICSIndustry code — you'll trim to 2 digitsApprovalFYYear the loan was approvedTermLoan length in monthsNoEmpNumber of employeesNewExist1 = existing business, 2 = new businessCreateJobNumber of jobs the loan was supposed to createRetainedJobNumber of jobs retainedFranchiseCode0 or 1 = not a franchise, anything else = is a franchiseUrbanRural1 = urban, 2 = ruralRevLineCrRevolving credit line — Y or NLowDocFast track loan program — Y or NDisbursementGrossActual dollar amount given outGrAppvDollar amount the bank approvedSBA_AppvDollar amount the SBA guaranteedMIS_StatusThe outcome — CHGOFF or P I F
1.2 Columns We Drop and Why
ColumnWhy It Gets DroppedLoanNr_ChkDgtJust an ID number, useless for MLNameBorrower name, useless for MLCityToo granular, State is enoughZipToo granular, State is enoughBankNot relevant to our storyBankStateNot relevant to our storyDisbursementDateTiming info we don't needApprovalDateApprovalFY already captures the yearChgOffDateDANGER — only exists after a default, leaks the answerChgOffPrinGrDANGER — same problem, drop immediatelyBalanceGrossReflects what happened after the loan, not before

2. The Target Variable
2.1 What It Is
The thing we are predicting. One column, two values, created by Person B during cleaning.
2.2 How To Create It
python# Step 1 — keep only rows with a real completed outcome
df = df[df['MIS_Status'].isin(['CHGOFF', 'P I F'])]
# Note: PIF has spaces in it — 'P I F' not 'PIF'

# Step 2 — convert to binary number
df['default'] = (df['MIS_Status'] == 'CHGOFF').astype(int)
# 1 = loan defaulted (bad outcome)
# 0 = loan paid back in full (good outcome)
2.3 What The Numbers Mean

1 = the business stopped paying, the loan was charged off
0 = the business paid the loan back completely


3. The Cleaned Dataframe Contract
3.1 What This Is
This is the agreement between Person B (who cleans) and Person A (who models). When Person B hands over df_clean.csv at noon Sunday, it must look exactly like this. No exceptions.
3.2 Every Column and Its Expected Format
ColumnTypeExample ValuesNotesStatestring'NY', 'CA', 'TX'Sectorstring'72', '44', '52'First 2 digits of NAICSApprovalFYinteger2005, 2008, 2012Terminteger84, 120, 240NoEmpinteger5, 12, 200NewExistinteger1 or 2 onlyCreateJobinteger0, 3, 10RetainedJobinteger0, 5, 20IsFranchiseinteger0 or 1 onlyCleaned from FranchiseCodeUrbanRuralinteger1 or 2 onlyDrop rows where value is 0RevLineCrinteger0 or 1 onlyCleaned from Y/NLowDocinteger0 or 1 onlyCleaned from Y/NDisbursementGrossfloat150000.0Dollar signs removedGrAppvfloat175000.0Dollar signs removedSBA_Appvfloat87500.0Dollar signs removeddefaultinteger0 or 1 onlyThe target variable
3.3 Rules For The Handoff

No null values anywhere
No dollar signs in any column
No string values except State and Sector
No columns outside the list above
Saved as data/df_clean.csv


4. Engineered Features
4.1 What These Are
New columns Person A creates from the cleaned dataframe. These are built on top of what Person B hands over. Person B does not need to build these — just know they're coming.
4.2 Every Feature, Its Code, and What It Means
loan_to_jobs_ratio
pythondf['loan_to_jobs_ratio'] = df['DisbursementGross'] / (df['CreateJob'] + 1)
What it means: How much did this loan cost per job it was supposed to create. High cost per job might mean the business isn't very productive. The +1 stops us dividing by zero when no jobs were created.

is_recession
pythondf['is_recession'] = df['ApprovalFY'].isin([2008, 2009]).astype(int)
What it means: Was this loan approved during the financial crisis. Loans from 2008 and 2009 behaved completely differently from normal years and the model needs to know that.

industry_default_rate
pythondf['industry_default_rate'] = df.groupby('Sector')['default'].transform('mean')
What it means: How risky is this industry on average historically. A restaurant applying for a loan is in a higher-risk sector than a consulting firm. This bakes that context into the model.

is_new_business
pythondf['is_new_business'] = (df['NewExist'] == 2).astype(int)
What it means: Is this a brand new business or an existing one. New businesses default at much higher rates so we make this explicit as its own feature.

sba_coverage_ratio
pythondf['sba_coverage_ratio'] = df['SBA_Appv'] / df['GrAppv']
What it means: What fraction of the loan did the SBA actually guarantee. Higher coverage means the government thought it was worth backing. Low coverage might signal the bank was less confident.

state_default_rate
pythondf['state_default_rate'] = df.groupby('State')['default'].transform('mean')
What it means: How risky is this state on average historically. Same idea as industry default rate but for geography. This is the feature that powers our fairness argument.

loan_vs_industry_avg
pythondf['loan_vs_industry_avg'] = df['DisbursementGross'] / df.groupby('Sector')['DisbursementGross'].transform('mean')
What it means: Is this loan unusually large for its industry. A $2M loan for a small restaurant is very different from a $2M loan for a manufacturing company.

disbursement_ratio
pythondf['disbursement_ratio'] = df['DisbursementGross'] / df['GrAppv']
```
**What it means:** What percentage of the approved amount was actually given out. If the bank approved $200k but only gave $150k, something happened between approval and disbursement.

---

# 5. Saturday Night Checklist

## 5.1 The Three Things You Confirm Out Loud Together

Go through these one by one. Both people say yes before anyone opens a laptop.

**Confirm 1 — Target variable:**
```
df['default'] — 1 = CHGOFF, 0 = P I F
```

**Confirm 2 — Cleaned dataframe columns:**
```
State, Sector, ApprovalFY, Term, NoEmp, NewExist,
CreateJob, RetainedJob, IsFranchise, UrbanRural,
RevLineCr, LowDoc, DisbursementGross, GrAppv, SBA_Appv, default
```

**Confirm 3 — Engineered features Person A will build:**
```
loan_to_jobs_ratio, is_recession, industry_default_rate,
is_new_business, sba_coverage_ratio, state_default_rate,
loan_vs_industry_avg, disbursement_ratio
5.2 The Rule
Once you both confirm all three you split off and work independently. You should not need to check in about structure at any point after this conversation. If you're asking each other what columns exist, this checklist failed.