# 🚀 GETTING STARTED - Federated Learning Credit Risk Project

Welcome to the **Federated Learning for Credit Risk Assessment** repository.

> [!IMPORTANT]
> **Primary Branch**: The primary branch for this repository is **`v1.0-submission-snapshot`**. Always make sure you clone or check out this branch.

---

## 📁 Repository Structure

```
federated-credit-risk-assessment/
├── 📄 README.md                           # Main project overview & documentation
├── 📄 LICENSE                             # MIT License
├── 📄 requirements.txt                    # Python dependencies
├── 📂 data/
│   ├── raw/                               # Raw dataset instructions
│   └── processed/                         # Cleaned & scaled datasets (v1, v2, v3)
├── 📂 src/
│   └── data/                              # Source data loading modules
├── 📂 utils/
│   ├── preprocessing.py                   # Data cleaning & SMOTE pipeline
│   ├── evaluation.py                      # Metrics calculation & report generation
│   ├── visualization.py                   # Plotting helper functions
│   └── data_loader.py                     # Data loader utility
├── 📂 models/
│   ├── centralized/                       # Logistic Regression, Random Forest, NN
│   └── federated/                         # Client, Server, and Coordinator modules
├── 📂 experiments/
│   ├── compare_models.py                  # Baseline comparative execution
│   └── federated_training.py              # Main FedAvg experiment runner
├── 📂 notebooks/
│   └── 01_data_exploration.ipynb          # Exploratory Data Analysis notebook
├── 📂 docs/                               # Phase guides, timeline, and documentation
├── 📂 visualization/                      # Summary visualization charts
└── 📂 results/                            # Metrics JSONs, ROC/PR plots, and FL logs
```

---

## ⚡ Quick Start

### Step 1: Clone Repository & Install Dependencies

```bash
# Clone the repository using the submission branch
git clone -b v1.0-submission-snapshot https://github.com/Samuel-025/federated-credit-risk-assessment.git

# Navigate to project folder
cd federated-credit-risk-assessment

# Install required packages
pip install -r requirements.txt
```

### Step 2: Run Centralized Baselines

```bash
# Navigate to models/centralized directory
cd models/centralized

# Train individual baseline models
python logistic_regression_train.py
python random_forest_train.py
python neural_network_train.py

# Return to project root
cd ../..
```

### Step 3: Run Baseline Model Comparison

```bash
# Navigate to experiments directory and run comparison
cd experiments
python compare_models.py

# Return to project root
cd ..
```

### Step 4: Run Federated Learning Experiment

```bash
# Launch FedAvg federated training simulation from project root
python experiments/federated_training.py
```

### Step 5: Explore EDA Notebook (Optional)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 📖 Key Documentation Files

1. `README.md` - High-level project summary, methodology, results, license, and topics.
2. `QUICK_REFERENCE.md` - Handy reference card for commands, scripts, and code snippets.
3. `PHASE1_COMPLETE.md` & `PHASE2_COMPLETE.md` - Detailed breakdowns of Phase 1 and Phase 2 implementations.
4. `models/centralized/README.md` - Guide to centralized models.
5. `models/federated/README.md` - Guide to the federated framework (Server, Client, Coordinator).

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Git branch check
To confirm you are on the correct branch:
```bash
git branch
```
Output should display `* v1.0-submission-snapshot`.

---

## 📄 License
Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
