# 📋 PHASE 2 QUICK REFERENCE

**Phase:** Preprocessing & Baseline Models  
**Duration:** 10-14 days  
**Goal:** Train 3 models, benchmark performance

---

## ⚡ QUICK COMMANDS

### Preprocessing (Day 1)
```python
from utils.data_loader import load_german_credit_data
from utils.preprocessing import CreditDataPreprocessor, save_processed_data

df = load_german_credit_data()
prep = CreditDataPreprocessor()
result = prep.preprocess_pipeline(df, encoding='label', scaling='standard', 
                                    feature_eng=True, balance_method='smote')
save_processed_data(result, prefix='v2_standard_smote')
```

### Logistic Regression (Day 2)
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X_train = np.load('data/processed/v2_standard_smote_X_train.npy')
y_train = np.load('data/processed/v2_standard_smote_y_train.npy')

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```

### Random Forest (Day 3)
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
```

### Neural Network (Day 4-5)
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=23),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## 📊 EXPECTED RESULTS

| Model | Accuracy | F1 | AUC | Time |
|-------|----------|----|----|------|
| Logistic Regression | 72-75% | 0.70 | 0.75 | <1s |
| Random Forest | 75-78% | 0.73 | 0.78 | 10s |
| Neural Network | 76-80% | 0.74 | 0.80 | 5min |

---

## 🗂️ FILE STRUCTURE

```
data/processed/
├── v1_standard_nobalance_*.npy  # For Random Forest
├── v2_standard_smote_*.npy      # Main (for LR)
└── v3_minmax_smote_*.npy        # For Neural Net

models/centralized/
├── logistic_regression.pkl
├── random_forest.pkl
└── neural_network.h5

results/centralized/
├── *_metrics.json
├── *_confusion_matrix.png
└── model_comparison.csv
```

---

## ✅ DAILY CHECKLIST

**Day 1:** 
- [ ] Run preprocessing V2
- [ ] Verify files created
- [ ] Check data shapes

**Day 2:**
- [ ] Train Logistic Regression
- [ ] Accuracy >72%?
- [ ] Save model

**Day 3:**
- [ ] Train Random Forest
- [ ] Accuracy >75%?
- [ ] Save model

**Day 4-5:**
- [ ] Install TensorFlow
- [ ] Train Neural Network
- [ ] Accuracy >76%?

**Day 6:**
- [ ] Compare all models
- [ ] Create comparison table
- [ ] Save visualization

---

## 🔧 TROUBLESHOOTING

**Low accuracy (<70%)**
→ Check SMOTE applied, verify preprocessing

**"Module not found"**
→ Run from project root directory

**Neural Network not converging**
→ Use V3 data (MinMax scaled)

**SMOTE too slow**
→ Use `sampling_strategy=0.8`

---

## 📝 FOR BLACKBOOK

**Chapter 6.1: Preprocessing** (3 pages)
- Encoding strategy
- Scaling methods
- SMOTE explanation

**Chapter 6.2: Models** (9 pages)
- 6.2.1: Logistic Regression
- 6.2.2: Random Forest  
- 6.2.3: Neural Network

Each model: algorithm + hyperparameters + results

---

## 🎯 SUCCESS CRITERIA

✅ All 3 models trained  
✅ Best model >75% accuracy  
✅ Comparison table created  
✅ Chapter 6 drafted (10+ pages)  
✅ All plots saved  

---

## 📅 TIMELINE

- **Days 1-2:** Preprocessing
- **Days 3-5:** Train models
- **Days 6-7:** Compare & analyze
- **Days 8-14:** Write Chapter 6

---

## 🚀 QUICK START

```bash
# 1. Preprocessing
python -c "from utils.preprocessing import *; ..."

# 2. Train models
python models/centralized/logistic_regression_train.py
python models/centralized/random_forest_train.py
python models/centralized/neural_network_train.py

# 3. Compare
python experiments/compare_models.py
```

---

**Ready?** Start with preprocessing! See PHASE2_QUICKSTART.md for details.
