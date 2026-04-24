"""
Logistic Regression - Baseline Model
Copy and run this file to train Logistic Regression
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append('../..')

from utils.evaluation import calculate_metrics, print_metrics, save_evaluation_report

print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION")
print("="*70)

# 1. Load preprocessed data (V2 - with SMOTE)
print("\n1. Loading data...")
X_train = np.load('../../data/processed/v2_standard_smote_X_train.npy')
y_train = np.load('../../data/processed/v2_standard_smote_y_train.npy')
X_test = np.load('../../data/processed/v2_standard_smote_X_test.npy')
y_test = np.load('../../data/processed/v2_standard_smote_y_test.npy')

print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")

# 2. Train model
print("\n2. Training Logistic Regression...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    solver='lbfgs'
)

model.fit(X_train, y_train)
print("✓ Model trained")

# 3. Evaluate on test set
print("\n3. Evaluating on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
print_metrics(metrics, "Logistic Regression")

# 4. Save model
print("\n4. Saving model...")
Path('../../models/centralized').mkdir(parents=True, exist_ok=True)
joblib.dump(model, '../../models/centralized/logistic_regression.pkl')
print("✓ Model saved to models/centralized/logistic_regression.pkl")

# 5. Save evaluation report
Path('../../results/centralized').mkdir(parents=True, exist_ok=True)
save_evaluation_report(
    y_test, y_pred, y_pred_proba,
    model_name="LogisticRegression",
    save_dir='../../results/centralized'
)

print("\n" + "="*70)
print("✅ LOGISTIC REGRESSION COMPLETE")
print("="*70)
print(f"\nAccuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"AUC-ROC: {metrics.get('roc_auc', 0):.4f}")