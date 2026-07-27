# Federated Learning for Credit Risk Assessment

A Privacy-Preserving Approach to Loan Default Prediction

**Final Year Project - BSc Data Science**  
**Mumbai University | Academic Year: 2024-2025**  
**Status: ✅ Project Completed & Submitted**

---

> [!IMPORTANT]
> **Primary Branch Notice**  
> The primary development and submission branch for this repository is **`v1.0-submission-snapshot`**. All analysis, reproduction, and contributions should strictly use this branch.

---

## 📋 Project Overview
This project implements a **Federated Learning (FL)** framework for credit risk assessment. Using the Federated Averaging (**FedAvg**) algorithm, we demonstrate how financial institutions can collaboratively train a robust machine learning model for loan default prediction without exposing sensitive customer data.

### Key Accomplishments
1. **Hybrid Architecture**: Built a modular system supporting both centralized baselines and federated simulations.
2. **Performance Parity**: Achieved predictive performance within 2% of centralized models while maintaining data privacy.
3. **Robustness Testing**: Evaluated model performance across both **IID** (Independent and Identically Distributed) and **Non-IID** data partitions.
4. **Comprehensive Analysis**: Performed convergence studies, feature importance analysis, and loss/accuracy tracking across communication rounds.

---

## 🔬 Methodology & Final Results

### Phase 1: Data Preparation & Preprocessing ✅
* **Dataset**: German Credit Dataset (1,000 samples, 20 original features).
* **Preprocessing**: Feature engineering (e.g., credit-to-duration ratios), categorical encoding, scaling (Standard & MinMax), and class balancing (SMOTE).

### Phase 2: Centralized Baseline Models ✅
Established benchmarks using traditional machine learning models trained on consolidated data:
* **Logistic Regression**: Accuracy 73.5%, F1-Score 0.71, AUC-ROC 0.77.
* **Random Forest**: Accuracy 76.8%, F1-Score 0.75, AUC-ROC 0.80.
* **Neural Network**: Accuracy 78.2%, F1-Score 0.76, AUC-ROC 0.81 (Best Centralized Baseline).

### Phase 3: Federated Learning Implementation ✅
* **Algorithm**: Federated Averaging (FedAvg).
* **Environment**: 10 simulated client nodes with IID and Non-IID data distributions.
* **Results**: Global model converged within **15 communication rounds**, achieving an accuracy of ~76.5% under IID conditions (within 2% of the centralized Neural Network).

### Phase 4: Final Evaluation & Visualizations ✅
* **Convergence Analysis**: Verified stable convergence under client data heterogeneity.
* **Visual Artifacts**: Generated ROC curves, PR curves, feature importance charts, and FL loss/accuracy plots.

---

## 🗂️ Project Structure

```
federated-credit-risk-assessment/
├── LICENSE                                # MIT License
├── README.md                              # Main project overview & documentation
├── GETTING_STARTED.md                     # Detailed onboarding guide
├── QUICK_REFERENCE.md                     # Fast reference card for commands & scripts
├── PHASE1_COMPLETE.md                     # Phase 1 completion summary
├── PHASE2_COMPLETE.md                     # Phase 2 completion summary
├── PHASE2_QUICK_REFERENCE.md               # Quick guide for Phase 2 baselines
├── PHASE2_SETUP_COMPLETE.md               # Preprocessing & baseline setup report
├── PHASE2_STARTER.py                      # Baseline starter script
├── PHASE_1_REPORT.txt                     # Summary report for Phase 1
├── QUICK_START_GUIDE.txt                  # Quick start guide text version
├── requirements.txt                       # Project dependencies
├── code snippets.py                       # Helpful utility snippets
├── preprocessing.ipynb                    # Interactive preprocessing notebook
├── data/
│   ├── raw/                               # Raw dataset instructions & files
│   └── processed/                         # Processed datasets (v1, v2, v3) & metadata
├── docs/                                  # Project guides, timelines & blackbook notes
├── experiments/
│   ├── compare_models.py                  # Baseline comparison script
│   └── federated_training.py              # Main FL experiment launcher
├── models/
│   ├── centralized/                       # Centralized ML model training scripts
│   │   ├── logistic_regression_train.py
│   │   ├── random_forest_train.py
│   │   └── neural_network_train.py
│   └── federated/                         # Federated learning components
│       ├── client.py                      # FL client node simulation
│       ├── coordinator.py                 # FL workflow coordinator
│       └── server.py                      # FL server & FedAvg aggregator
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Interactive EDA notebook
│   └── 01_EDA_and_Preprocessing.py        # EDA python module
├── results/
│   ├── eda_summary_statistics.csv         # Summary stats
│   ├── centralized/                       # Metrics, ROC/PR curves & matrices
│   ├── eda_plots/                         # Exploratory data analysis plots
│   └── federated/                         # FL convergence logs & metrics
├── src/
│   └── data/                              # Source data loader module
├── utils/                                 # Preprocessing, evaluation & plotting tools
└── visualization/                         # Generated summary charts
```

---

## ⚡ Quick Start & Setup

### 1. Clone the Repository (Branch: `v1.0-submission-snapshot`)
```bash
git clone -b v1.0-submission-snapshot https://github.com/Samuel-025/federated-credit-risk-assessment.git
cd federated-credit-risk-assessment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Centralized Baselines
```bash
# Navigate to models/centralized directory and run training scripts
cd models/centralized
python logistic_regression_train.py
python random_forest_train.py
python neural_network_train.py
cd ../..
```

### 4. Run Baseline Comparison
```bash
# Navigate to experiments directory and run comparison
cd experiments
python compare_models.py
cd ..
```

### 5. Run Federated Learning Experiments
```bash
# Run FL experiment launcher from project root
python experiments/federated_training.py
```

---

## 📚 Key Technologies
* **Language**: Python 3.8+
* **Machine Learning**: Scikit-learn, Imbalanced-learn (SMOTE)
* **Deep Learning**: TensorFlow / Keras
* **Visualization**: Matplotlib, Seaborn
* **Methodology**: Federated Averaging (FedAvg), Differential Privacy (Theoretical Framework)

---

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

---

## 👤 Authors
* **Suyash Madke** — BSc Data Science, University of Mumbai
* **Aditya Veer** — BSc Data Science, University of Mumbai

*Last Updated: July 2026 (Documentation & Branch Refinement)*
