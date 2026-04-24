# Federated Learning for Credit Risk Assessment

A Privacy-Preserving Approach to Loan Default Prediction

**Final Year Project - BSc Data Science** **Mumbai University | Academic Year: 2024-2025** **Status: ✅ Project Completed & Submitted**

---

## 📋 Project Overview
This project successfully implemented a **Federated Learning (FL)** framework for credit risk assessment. By utilizing the FedAvg algorithm, we demonstrated how financial institutions can collaboratively train a robust machine learning model for loan default prediction without exposing sensitive customer data.

### Key Accomplishments
1. **Hybrid Architecture**: Developed a modular system supporting both centralized baselines and federated simulation.
2. **Performance Parity**: Achieved results comparable to centralized training while maintaining data decentralization.
3. **Robustness Testing**: Evaluated model performance across **IID** (Independent and Identically Distributed) and **Non-IID** data partitions.
4. **Comprehensive Analysis**: Conducted convergence studies and privacy-utility tradeoff assessments.

---

## 🔬 Final Methodology & Results

### Phase 1: Data Preparation ✅
* **Dataset**: Utilized a synthetic version of the German Credit Dataset (1,000 samples).
* **Preprocessing**: Implemented full feature engineering, scaling, and class balancing (SMOTE).

### Phase 2: Centralized Baselines ✅
We established baseline metrics using traditional ML models to compare against the federated approach:
* **Models**: Logistic Regression, Random Forest, and a 3-layer Neural Network.
* **Outcome**: Random Forest provided the strongest baseline for individual data silos.

### Phase 3: Federated Learning Implementation ✅
* **Algorithm**: Federated Averaging (FedAvg).
* **Setup**: 10 clients with varying data distributions.
* **Results**: The Global Model stabilized within **15 communication rounds**, reaching an accuracy within 2% of the centralized Neural Network.

### Phase 4: Final Evaluation ✅
* **Convergence**: Validated that the global model converges even under Non-IID conditions.
* **Visuals**: Generated ROC curves, Precision-Recall curves, and loss/accuracy history for the final report.

---

## 🗂️ Project Structure
```
federated_credit_risk/
├── data/               # Preprocessed and partitioned datasets
├── models/             # Saved .pkl and .h5 model files
├── notebooks/          # Step-by-step EDA and training logic
├── results/            # Final metrics, confusion matrices, and plots
├── utils/              # Helper scripts for FL and evaluation
└── requirements.txt    # Project dependencies
```

---

## 📈 Final Progress Tracker

| Phase | Task | Status | Completion |
|-------|------|--------|------------|
| 1 | Data Preparation | ✅ Complete | 100% |
| 2 | Centralized Models | ✅ Complete | 100% |
| 3 | FL Implementation | ✅ Complete | 100% |
| 4 | Evaluation & Analysis | ✅ Complete | 100% |
| 5 | Submission & Documentation | ✅ Complete | 100% |

**Overall Progress: 100%**

---

## 📚 Key Technologies
* **Language**: Python 3.8+
* **ML/DL**: Scikit-learn, TensorFlow
* **Visualization**: Matplotlib, Seaborn
* **Methodology**: Federated Averaging (FedAvg), Differential Privacy (Theory)

---

## 👤 Author
**[Your Name]** BSc Data Science, University of Mumbai  
**Project Guide**: [Professor's Name]

---

*Last Updated: April 24, 2026 (Final Submission Version)*
    
