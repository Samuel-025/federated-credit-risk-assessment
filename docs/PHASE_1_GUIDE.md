# Phase 1: Data Preparation - Quick Start Guide

## Week 1-2 Tasks Checklist

### Week 1: Setup & Data Collection

#### Day 1-2: Environment Setup
- [ ] Install Python 3.8+ (if not already installed)
- [ ] Create virtual environment
  ```bash
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  ```
- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Verify installation
  ```bash
  python -c "import tensorflow; print(tensorflow.__version__)"
  python -c "import pandas; print('Success!')"
  ```

#### Day 3-4: Data Download
- [ ] Download German Credit dataset
  - Visit: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
  - Save to: `data/raw/german_credit/german.data`
  - Download documentation too: `german.doc`
  
- [ ] (Optional) Download Lending Club dataset
  - Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club
  - Requires Kaggle account
  - Save to: `data/raw/lending_club/`
  - **Note:** This is 2GB+, use for final experiments

#### Day 5-7: Initial Exploration
- [ ] Test data loader
  ```bash
  cd src/data
  python data_loader.py
  ```
- [ ] Run exploratory notebook
  ```bash
  jupyter notebook notebooks/01_data_exploration.ipynb
  ```
- [ ] Complete all cells in the notebook
- [ ] Document key findings

### Week 2: Preprocessing & Partitioning

#### Day 1-3: Data Preprocessing
- [ ] Create preprocessing module
- [ ] Handle missing values (if any)
- [ ] Encode categorical features
- [ ] Scale numerical features
- [ ] Feature engineering (create new features)
- [ ] Document preprocessing decisions

#### Day 4-5: Federated Data Partitioning
- [ ] Create federated partitioner module
- [ ] Split data into N "banks" (recommend 3-5)
- [ ] Create IID (Independent Identically Distributed) splits
- [ ] Create non-IID splits (realistic scenario)
- [ ] Verify partition quality
- [ ] Save partitioned datasets

#### Day 6-7: Validation & Documentation
- [ ] Run preprocessing notebook
- [ ] Verify data quality after preprocessing
- [ ] Document preprocessing pipeline
- [ ] Save processed data
- [ ] Update project log/diary

---

## Detailed Instructions

### 1. Environment Setup (Days 1-2)

**Create Project Directory:**
```bash
mkdir ~/federated_credit_risk
cd ~/federated_credit_risk
```

**Clone/Download Project Structure:**
(Use the files I've created for you)

**Create Virtual Environment:**
```bash
# Create venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Verify TensorFlow and TF Federated:**
```bash
python -c "import tensorflow as tf; import tensorflow_federated as tff; print(f'TF: {tf.__version__}, TFF: {tff.__version__}')"
```

If you get errors with TensorFlow Federated, try:
```bash
pip install tensorflow==2.13.0
pip install tensorflow-federated==0.57.0
```

---

### 2. Data Download (Days 3-4)

**German Credit Dataset (Start Here):**

1. Visit: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
2. Click "Download" button
3. Extract the zip file
4. You should get: `german.data` and `german.doc`
5. Create directory:
   ```bash
   mkdir -p data/raw/german_credit
   ```
6. Move files:
   ```bash
   mv german.data data/raw/german_credit/
   mv german.doc data/raw/german_credit/
   ```

**Verify Download:**
```bash
python src/data/data_loader.py
```

You should see:
```
German Credit Data Loaded:
  - Records: 1000
  - Features: 20
  - Target distribution: {1: 700, 0: 300}
```

---

### 3. Exploratory Data Analysis (Days 5-7)

**Launch Jupyter Notebook:**
```bash
jupyter notebook
```

**Navigate to:** `notebooks/01_data_exploration.ipynb`

**Complete These Sections:**
1. Load dataset ✓
2. Basic statistics ✓
3. Missing value analysis ✓
4. Target distribution ✓
5. Feature distributions ✓
6. Correlation analysis ✓
7. Outlier detection ✓

**Key Things to Note:**
- Class imbalance ratio
- Which features are most correlated with target
- Any data quality issues
- Feature types (numerical vs categorical)

**Save Key Plots:**
Save important visualizations to `results/figures/eda/`:
- Target distribution
- Correlation heatmap
- Key feature distributions
- Box plots for outliers

---

### 4. Data Preprocessing (Week 2, Days 1-3)

**Create Preprocessing Pipeline:**

I'll help you create this in the next phase, but here's what it will include:

1. **Missing Value Handling:**
   - Numerical: Mean/Median imputation or KNN imputer
   - Categorical: Mode imputation or create 'Unknown' category

2. **Categorical Encoding:**
   - One-hot encoding for nominal features
   - Label encoding for ordinal features
   - Consider target encoding for high-cardinality

3. **Numerical Scaling:**
   - StandardScaler (mean=0, std=1) for most models
   - MinMaxScaler (0-1) for neural networks
   - RobustScaler for features with outliers

4. **Feature Engineering:**
   - Create interaction features
   - Polynomial features (if needed)
   - Domain-specific features (e.g., credit_amount/duration ratio)

---

### 5. Federated Data Partitioning (Week 2, Days 4-5)

**Partitioning Strategy:**

Simulate 3-5 "banks" with different data:

**IID Partitioning (Equal distribution):**
- Randomly shuffle and split data equally
- Each bank gets similar class distribution
- Easier to train, baseline scenario

**Non-IID Partitioning (Realistic):**
- Each bank gets different class distributions
- Simulate real-world data heterogeneity
- More challenging, realistic scenario

**Example:**
```python
# Bank 1: 60% good, 40% bad
# Bank 2: 70% good, 30% bad  
# Bank 3: 75% good, 25% bad
# Bank 4: 65% good, 35% bad
# Bank 5: 80% good, 20% bad
```

---

## Expected Outputs by End of Phase 1

### Files Created:
```
data/
├── raw/
│   └── german_credit/
│       ├── german.data
│       └── german.doc
├── processed/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── eda_summary_statistics.csv
│   └── correlation_matrix.csv
└── federated_splits/
    ├── bank_1_train.csv
    ├── bank_1_test.csv
    ├── bank_2_train.csv
    ├── bank_2_test.csv
    ├── ... (for all banks)
    └── partition_info.json
```

### Notebooks Completed:
- [x] 01_data_exploration.ipynb
- [x] 02_preprocessing.ipynb (to be created)

### Documentation:
- [x] EDA findings documented
- [x] Preprocessing decisions recorded
- [x] Data quality report created

---

## Common Issues & Solutions

### Issue 1: TensorFlow Installation Errors
**Solution:**
```bash
pip install --upgrade pip
pip install tensorflow==2.13.0 --no-cache-dir
pip install tensorflow-federated==0.57.0
```

### Issue 2: Jupyter Not Starting
**Solution:**
```bash
pip install --upgrade jupyter
jupyter notebook --generate-config
jupyter notebook
```

### Issue 3: Dataset Download Issues
**Solution:**
- Use alternative UCI mirror
- Download manually via browser
- Check internet connection/firewall

### Issue 4: Import Errors
**Solution:**
```bash
# Verify you're in virtual environment
which python  # Should show venv path

# Reinstall in editable mode
pip install -e .
```

---

## Daily Progress Log Template

Keep track of your work:

```
## Week 1

### Day 1 - [Date]
**Tasks Completed:**
- 

**Issues Faced:**
- 

**Solutions:**
- 

**Tomorrow's Plan:**
- 

### Day 2 - [Date]
...
```

---

## Questions to Answer Before Moving to Phase 2

1. What is the class distribution in your dataset?
2. Are there any missing values? How did you handle them?
3. Which features are most correlated with the target?
4. What preprocessing steps did you apply?
5. How many "banks" did you create for federated simulation?
6. What's the data distribution across banks (IID vs non-IID)?
7. What's the train-test split ratio you used?

---

## Next Phase Preview

**Phase 2: Centralized Baseline Models**
- Implement Logistic Regression
- Implement Random Forest
- Implement Neural Network
- Evaluate all models
- Create baseline for comparison with FL

---

## Resources

**Documentation:**
- TensorFlow: https://www.tensorflow.org/
- TF Federated: https://www.tensorflow.org/federated
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/

**Research Papers to Read:**
1. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. "Federated Learning: Challenges, Methods, and Future Directions"
3. Any recent papers on FL in finance (I can search for these)

**Useful Tutorials:**
- TF Federated tutorials: https://www.tensorflow.org/federated/tutorials/tutorials_overview
- Credit scoring with ML: Search on Medium/TowardsDataScience

---

Good luck with Phase 1! Let me know when you're ready to move to Phase 2 or if you need help with any specific part.
