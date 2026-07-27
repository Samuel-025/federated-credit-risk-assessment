# 📋 Quick Reference Card - Federated Learning Credit Risk Project

---

## 🎯 Project At a Glance

**Title:** Federated Learning for Credit Risk Assessment  
**Type:** Research-based BSc Data Science Final Year Project  
**Primary Branch:** `v1.0-submission-snapshot`  
**Dataset:** German Credit (1000 samples, 20 original features)  
**Objective:** Compare centralized baseline models against Federated Learning (FedAvg) for credit scoring

---

## ⚡ Quick Commands

### Clone Repository (Submission Branch)
```bash
git clone -b v1.0-submission-snapshot https://github.com/Samuel-025/federated-credit-risk-assessment.git
cd federated-credit-risk-assessment
```

### Setup Dependencies
```bash
pip install -r requirements.txt
```

### Run Centralized Baselines
```bash
cd models/centralized
python logistic_regression_train.py
python random_forest_train.py
python neural_network_train.py
cd ../..
```

### Run Baseline Comparison
```bash
cd experiments
python compare_models.py
cd ..
```

### Run Federated Learning Simulation
```bash
python experiments/federated_training.py
```

---

## 📚 Key Files & Scripts

| File | Purpose |
|------|---------|
| `README.md` | Comprehensive project overview & license |
| `GETTING_STARTED.md` | Detailed setup & onboarding guide |
| `models/centralized/` | Centralized ML training scripts (LR, RF, NN) |
| `models/federated/` | FL Server, Client, & Coordinator modules |
| `experiments/federated_training.py` | Main FedAvg experiment launcher |
| `utils/preprocessing.py` | Feature engineering, scaling & SMOTE pipeline |
| `utils/evaluation.py` | Evaluation metrics calculation & chart generation |

---

## 📈 Performance Summary

| Model | Type | Accuracy | F1-Score | AUC-ROC | Status |
|-------|------|----------|----------|---------|--------|
| Logistic Regression | Centralized | 73.5% | 0.71 | 0.77 | ✅ Complete |
| Random Forest | Centralized | 76.8% | 0.75 | 0.80 | ✅ Complete |
| Neural Network | Centralized | 78.2% | 0.76 | 0.81 | ✅ Complete |
| **FedAvg Global Model** | **Federated** | **76.5%** | **0.74** | **0.79** | **✅ Complete** |

---

## 📄 License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
