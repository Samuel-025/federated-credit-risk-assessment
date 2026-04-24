# Dataset Sources and Download Instructions

## Primary Datasets for Credit Risk Analysis

### 1. German Credit Data (Recommended to Start)

**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

**Download Steps:**
1. Visit the URL above
2. Click "Download" button
3. Save `german.data` to `data/raw/german_credit/`
4. Also download `german.doc` for feature descriptions

**Dataset Details:**
- Records: 1,000
- Features: 20 (7 numerical, 13 categorical)
- Target: Credit risk (Good=1, Bad=2)
- Class distribution: 70% Good, 30% Bad

**Features:**
- Status of existing checking account
- Duration in months
- Credit history
- Purpose
- Credit amount
- Savings account/bonds
- Present employment since
- Installment rate in percentage of disposable income
- Personal status and sex
- Other debtors/guarantors
- Present residence since
- Property
- Age in years
- Other installment plans
- Housing
- Number of existing credits at this bank
- Job
- Number of people being liable to provide maintenance for
- Telephone
- Foreign worker

---

### 2. Lending Club Loan Data (For Scaling Up)

**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

**Download Steps:**
1. Create Kaggle account if you don't have one
2. Visit the URL above
3. Download `accepted_2007_to_2018Q4.csv.gz`
4. Extract and save to `data/raw/lending_club/`

**Dataset Details:**
- Records: 2+ million
- Features: 150+
- Target: loan_status (Fully Paid, Charged Off, Default, etc.)
- Time range: 2007-2018

**Key Features:**
- loan_amnt, funded_amnt
- term (36/60 months)
- int_rate, installment
- grade, sub_grade
- emp_length
- annual_inc
- dti (debt-to-income ratio)
- delinq_2yrs
- fico_range_low, fico_range_high
- revol_bal, revol_util
- And many more...

---

### 3. Home Credit Default Risk (Advanced/Optional)

**Source:** Kaggle Competition  
**URL:** https://www.kaggle.com/c/home-credit-default-risk/data

**Download Steps:**
1. Visit the URL above
2. Download all CSV files (7 files total)
3. Save to `data/raw/home_credit/`

**Main Files:**
- application_train.csv (307,511 rows)
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv

**Dataset Details:**
- Highly imbalanced (only 8% default)
- Multiple tables (relational structure)
- Real-world complexity

---

## Recommended Approach for This Project

### Phase 1: Prototype with German Credit
- Quick to download and process
- Perfect for building and testing your FL pipeline
- Good for initial experiments

### Phase 2: Scale to Lending Club
- More realistic and impressive for final report
- Larger dataset shows scalability
- Industry-standard dataset

### Phase 3 (Optional): Add Home Credit
- If time permits
- Demonstrates handling complex data structures
- Adds significant research value

---

## Data Storage Structure

After downloading, your `data/raw/` should look like:

```
data/raw/
├── german_credit/
│   ├── german.data
│   └── german.doc
├── lending_club/
│   └── accepted_2007_to_2018Q4.csv
└── home_credit/  (optional)
    ├── application_train.csv
    ├── application_test.csv
    └── ... (other files)
```

---

## Quick Start

1. Start with German Credit (smallest, easiest)
2. Run exploratory data analysis
3. Build preprocessing pipeline
4. Test with centralized models
5. Implement federated learning
6. Once everything works, repeat with Lending Club for final results

---

## Notes

- All datasets are publicly available and free
- Properly cite dataset sources in your blackbook
- German Credit is great for learning, Lending Club for final impressive results
- Keep raw data unchanged; always work with copies in `processed/`

---

## Alternative Datasets (If Needed)

If above datasets are inaccessible:
- **Give Me Some Credit** (Kaggle) - 150k records
- **Default of Credit Card Clients** (UCI) - 30k records, Taiwan data
- **South German Credit** (UCI) - Updated version of German Credit
