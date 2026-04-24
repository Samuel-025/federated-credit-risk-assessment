# Federated Learning for Credit Risk Assessment

A Privacy-Preserving Approach to Loan Default Prediction

**Final Year Project - BSc Data Science**  
**Mumbai University**  
**Academic Year: 2024-2025**

---

## 📋 Project Overview

This project explores the application of **Federated Learning** to credit risk assessment, enabling multiple financial institutions to collaboratively build accurate machine learning models without sharing sensitive customer data.

### Key Objectives
1. Develop a federated learning framework for credit default prediction
2. Compare performance: Federated vs Centralized learning approaches
3. Analyze privacy-utility tradeoffs in financial ML models
4. Evaluate communication efficiency and convergence in federated settings

---

## 🗂️ Project Structure

```
federated_credit_risk/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed data, train-test splits
│   └── federated_splits/       # Data partitioned for FL simulation
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.py
│   ├── 02_Centralized_Baseline.py
│   └── 03_Federated_Learning.py
│
├── models/
│   ├── centralized/            # Baseline models
│   └── federated/              # FL model implementations
│
├── utils/
│   ├── generate_synthetic_data.py
│   ├── data_loader.py
│   └── evaluation.py
│
├── experiments/
│   └── run_experiments.py
│
├── results/
│   ├── eda_plots/
│   ├── model_performance/
│   └── comparison_results/
│
└── visualization/
    └── plot_results.py
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone/download the project:
```bash
cd federated_credit_risk
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Project

**Phase 1: Data Preparation**
```bash
cd notebooks
python3 01_EDA_and_Preprocessing.py
```

**Phase 2: Centralized Baseline Models**
```bash
python3 02_Centralized_Baseline.py
```

**Phase 3: Federated Learning**
```bash
python3 03_Federated_Learning.py
```

---

## 📊 Dataset

**Synthetic Credit Risk Dataset** (1000 samples)

### Features (20 total):

**Numerical Features (7):**
- `duration_months`: Credit duration (3-72 months)
- `credit_amount`: Loan amount (500-20000)
- `installment_rate`: % of disposable income (1-4)
- `residence_since`: Years at current residence (1-4)
- `age`: Customer age (19-75)
- `existing_credits`: Number of existing credits (1-4)
- `num_dependents`: Number of dependents (0-2)

**Categorical Features (13):**
- `checking_status`: Account balance status (0-3)
- `credit_history`: Past credit behavior (0-4)
- `purpose`: Loan purpose (0-10)
- `savings_status`: Savings account status (0-4)
- `employment`: Employment duration (0-4)
- `personal_status`: Marital status and gender (0-4)
- `other_parties`: Co-applicants/guarantors (0-2)
- `property_magnitude`: Property ownership (0-3)
- `other_payment_plans`: Other installment plans (0-2)
- `housing`: Housing situation (0-2)
- `job`: Job skill level (0-3)
- `telephone`: Has telephone (0-1)
- `foreign_worker`: Foreign worker status (0-1)

**Target Variable:**
- `credit_risk`: 1 = Good Credit, 0 = Bad Credit
- Class distribution: 67.3% Good, 32.7% Bad

---

## 🔬 Methodology

### Phase 1: Data Preparation ✅
- [x] Dataset generation/collection
- [x] Exploratory Data Analysis
- [x] Feature engineering
- [x] Data preprocessing
- [x] Train-test splitting

### Phase 2: Centralized Baseline (In Progress)
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] Neural Network
- [ ] Performance evaluation

### Phase 3: Federated Learning Implementation
- [ ] Data partitioning (IID & Non-IID)
- [ ] FedAvg algorithm implementation
- [ ] Client-server architecture
- [ ] Local training loops
- [ ] Global model aggregation

### Phase 4: Evaluation & Analysis
- [ ] Performance comparison
- [ ] Privacy-utility tradeoff analysis
- [ ] Communication efficiency metrics
- [ ] Statistical significance testing

### Phase 5: Documentation
- [ ] Blackbook/Project Report
- [ ] Presentation (PPT)
- [ ] Code documentation

---

## 📈 Progress Tracker

| Phase | Task | Status | Completion |
|-------|------|--------|------------|
| 1 | Dataset Generation | ✅ Complete | 100% |
| 1 | EDA | ✅ Complete | 100% |
| 1 | Preprocessing | ✅ Complete | 100% |
| 2 | Centralized Models | 🔄 In Progress | 0% |
| 3 | FL Implementation | ⏳ Pending | 0% |
| 4 | Evaluation | ⏳ Pending | 0% |
| 5 | Documentation | ⏳ Pending | 0% |

**Overall Progress: 25%** (Phase 1 Complete)

---

## 📝 Phase 1 Deliverables (Completed)

✅ **Data Files:**
- `data/raw/credit_data.csv` (1000 samples)
- `data/processed/credit_data_preprocessed.csv`
- `data/processed/train_data.csv` (800 samples)
- `data/processed/test_data.csv` (200 samples)
- `data/processed/scaler.pkl`

✅ **Visualizations:** (6 plots in `results/eda_plots/`)
1. Target distribution
2. Age distribution by credit risk
3. Credit amount analysis
4. Correlation heatmap
5. Duration vs Amount scatter plot
6. Categorical features analysis

✅ **Code:**
- Data generation script
- EDA & preprocessing pipeline

---

## 📚 Key Technologies

- **Programming:** Python 3.8+
- **Data Science:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Deep Learning:** TensorFlow (upcoming)
- **Federated Learning:** TensorFlow Federated / PySyft (upcoming)

---

## 🎯 Expected Outcomes

1. Functioning federated learning system for credit risk
2. Comparative analysis: Centralized vs Federated performance
3. Privacy-utility tradeoff insights
4. Communication efficiency evaluation
5. Complete research report (Blackbook)
6. Presentation materials

---

## 📅 Timeline

| Week | Tasks |
|------|-------|
| 1-2 | ✅ Literature review, Data preparation |
| 3-4 | Centralized baseline models |
| 5-7 | Federated learning implementation |
| 8-9 | Experiments & evaluation |
| 10 | Blackbook writing |
| 11 | Presentation preparation |
| 12 | Final review & submission |

**Deadline:** Before April 2026

---

## 👤 Author

[Your Name]  
BSc Data Science  
Mumbai University  

**Project Guide:** [Professor Name]

---

## 📖 References

(To be added - 30-40 academic papers)

Key papers to include:
- McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Papers on FL in finance
- Privacy-preserving ML techniques
- Credit risk modeling research

---

## 📞 Contact

For queries regarding this project:
- Email: [your.email@example.com]
- GitHub: [your-github-username]

---

*Last Updated: February 14, 2026*
