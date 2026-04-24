# Phase 1: Data Preparation - Quick Start Guide

## 🎯 Objective
Download, explore, and understand the German Credit Dataset to prepare for federated learning experiments.

---

## ✅ What You'll Accomplish in Phase 1

1. ✓ Download German Credit Dataset
2. ✓ Understand dataset structure and features
3. ✓ Perform exploratory data analysis (EDA)
4. ✓ Identify preprocessing requirements
5. ✓ Document insights for blackbook

---

## 🚀 Step-by-Step Instructions

### Step 1: Set Up Your Environment

```bash
# Navigate to project directory
cd federated_credit_risk

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required packages
pip install --break-system-packages -r requirements.txt
```

### Step 2: Download the Dataset

```bash
# Run the data loader script
python utils/data_loader.py
```

**Expected Output:**
```
======================================================================
GERMAN CREDIT DATASET - DATA LOADER
======================================================================

Downloading German Credit Dataset...
✓ Dataset downloaded successfully!
✓ Saved to: data/raw/german_credit_raw.data
✓ Shape: (1000, 21)
✓ Columns: 21

======================================================================
GERMAN CREDIT DATASET INFORMATION
======================================================================

📊 Shape: 1000 rows × 21 columns
...
```

### Step 3: Run Exploratory Data Analysis

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What to do in the notebook:**
1. Run all cells sequentially (Shift + Enter)
2. Observe visualizations and statistics
3. Read the insights and recommendations
4. Take notes for your blackbook

**Key sections to focus on:**
- Target variable distribution (class imbalance)
- Numerical feature distributions
- Categorical feature frequencies
- Feature-target relationships
- Correlation analysis

### Step 4: Document Your Findings

Create a document with your Phase 1 findings:

**Template:**
```
PHASE 1 - DATA EXPLORATION FINDINGS
====================================

Dataset Overview:
- Name: German Credit Dataset
- Source: UCI ML Repository
- Size: 1000 samples, 20 features + 1 target
- Features: 7 numerical, 13 categorical
- Target: Binary (Good/Bad credit)

Class Distribution:
- Good credit (1): 700 samples (70%)
- Bad credit (2): 300 samples (30%)
- Imbalance ratio: 2.33:1

Key Insights:
1. [Add your observations]
2. [Add your observations]
3. [Add your observations]

Data Quality:
- Missing values: None ✓
- Duplicates: None ✓
- Outliers: Present in 'duration' and 'credit_amount'

Preprocessing Requirements:
1. Encode categorical features
2. Scale numerical features
3. Handle class imbalance
4. Feature engineering opportunities
```

---

## 📊 Expected Outputs from Phase 1

### Files Generated:
1. `data/raw/german_credit_raw.data` - Raw dataset
2. `data/raw/german_credit.csv` - CSV version
3. `results/eda_summary_statistics.csv` - Statistical summary

### Deliverables for Blackbook:
1. **Dataset Description section (Chapter 5)**
   - Source and justification
   - Feature descriptions table
   - Statistical summary
   
2. **EDA visualizations (for Results chapter)**
   - Class distribution plots
   - Feature distribution histograms
   - Correlation heatmap
   - Boxplots for outliers

3. **Key insights list**
   - Data quality assessment
   - Feature importance indicators
   - Preprocessing recommendations

---

## 🎨 Visualizations to Include in Blackbook

From the EDA notebook, save these plots for your report:

1. **Class Distribution** (Bar + Pie chart)
   - Shows 70-30 split
   
2. **Numerical Feature Distributions** (Histograms)
   - Duration, Credit Amount, Age, etc.
   
3. **Correlation Matrix** (Heatmap)
   - Shows relationships between features
   
4. **Feature vs Target** (Boxplots)
   - Shows how features differ by class
   
5.  
   - Checking status, Credit history, etc.

**How to save plots:**
```python
# In Jupyter, after creating a plot:
plt.savefig('../visualization/class_distribution.png', dpi=300, bbox_inches='tight')
```

---

## 📝 Writing Tips for Dataset Description (Blackbook Chapter 5)

### 5.1 Dataset Selection and Sources (2 pages)

```
The German Credit Dataset was selected from the UCI Machine Learning 
Repository for this research due to several compelling reasons:

1. Established Benchmark: This dataset is widely used in credit risk 
   modeling research, allowing for comparison with existing literature.

2. Appropriate Size: With 1,000 instances, it is suitable for simulating
   federated learning across multiple institutions while maintaining 
   statistical significance.

3. Realistic Features: The dataset contains 20 features commonly used in
   actual credit assessment, including account status, credit history,
   and loan purpose.

4. Binary Classification: The clear good/bad credit distinction aligns
   with practical credit risk assessment scenarios.

The dataset can be accessed at:
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

[Continue with more details about why this dataset is appropriate...]
```

### 5.2 Data Characteristics (2-3 pages)

Create a comprehensive table:

| Feature | Type | Description | Range/Values | Missing |
|---------|------|-------------|--------------|---------|
| checking_status | Categorical | Status of checking account | A11-A14 | 0 |
| duration | Numerical | Loan duration in months | 4-72 | 0 |
| ... | ... | ... | ... | ... |

Add statistical summaries and class distribution details.

### 5.3 Exploratory Data Analysis (2-3 pages)

Include:
- Distribution plots with interpretations
- Correlation analysis findings
- Class imbalance discussion
- Outlier detection results

---

## ⏰ Time Estimate: 2-3 Days

- **Day 1:** Setup + Download + Initial EDA (3-4 hours)
- **Day 2:** Deep dive EDA + Visualization (4-5 hours)
- **Day 3:** Documentation + Screenshots for blackbook (2-3 hours)

---

## ✅ Phase 1 Completion Checklist

Before moving to Phase 2, ensure you have:

- [ ] Downloaded dataset successfully
- [ ] Run all cells in EDA notebook
- [ ] Understood all 20 features
- [ ] Identified class imbalance (70-30)
- [ ] Generated all required visualizations
- [ ] Saved plots for blackbook
- [ ] Documented key findings
- [ ] Written initial draft of Dataset Description chapter
- [ ] Identified preprocessing requirements

---

## 🚨 Common Issues & Solutions

### Issue 1: Dataset download fails
**Solution:**
```python
# Manual download alternative
import urllib.request
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
urllib.request.urlretrieve(url, "data/raw/german_credit_raw.data")
```

### Issue 2: Jupyter kernel crashes
**Solution:**
- Reduce plot resolution: `plt.figure(figsize=(8, 6))` instead of (16, 12)
- Run cells one by one instead of "Run All"

### Issue 3: Import errors
**Solution:**
```bash
# Reinstall specific packages
pip install --break-system-packages pandas numpy matplotlib seaborn
```

---

## 📚 Additional Resources

### German Credit Dataset Documentation:
- UCI Repository: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
- Feature descriptions: Included in `utils/data_loader.py`

### Python Libraries Documentation:
- pandas: https://pandas.pydata.org/docs/
- matplotlib: https://matplotlib.org/stable/contents.html
- seaborn: https://seaborn.pydata.org/

### Research Papers (for literature review):
1. "German Credit Data Analysis" (search on Google Scholar)
2. "Credit Scoring using Machine Learning" (multiple papers available)
3. "Class Imbalance in Credit Risk" (for methodology section)

---

## 🎯 Next Steps

After completing Phase 1, you'll move to:

**Phase 2: Data Preprocessing & Feature Engineering**
- Encoding categorical variables
- Scaling numerical features
- Handling class imbalance
- Creating federated data splits
- Feature engineering

---

**Questions?** Document them as you go - they'll be useful for your presentation Q&A preparation!

**Good luck with Phase 1! 🚀**
