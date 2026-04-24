"""
Random Forest - Baseline Model
Copy and run this file to train Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append('../..')

from utils.evaluation import calculate_metrics, print_metrics, save_evaluation_report

print("\n" + "="*70)
print("TRAINING RANDOM FOREST")
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
print("\n2. Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=20,            # Maximum tree depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,               # Use all CPU cores
    verbose=0
)

model.fit(X_train, y_train)
print("✓ Model trained")
print(f"  Number of trees: {model.n_estimators}")
print(f"  Max depth: {model.max_depth}")

# 3. Evaluate on test set
print("\n3. Evaluating on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
print_metrics(metrics, "Random Forest")

# 4. Feature importance
print("\n4. Top 10 Most Important Features:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"   {i+1}. Feature {idx}: {importances[idx]:.4f}")

# Save feature importance
feature_importance_df = pd.DataFrame({
    'feature_index': range(len(importances)),
    'importance': importances
}).sort_values('importance', ascending=False)
feature_importance_df.to_csv('../../results/centralized/rf_feature_importance.csv', index=False)

# 5. Save model
print("\n5. Saving model...")
Path('../../models/centralized').mkdir(parents=True, exist_ok=True)
joblib.dump(model, '../../models/centralized/random_forest.pkl')
print("✓ Model saved to models/centralized/random_forest.pkl")

# 6. Save evaluation report
Path('../../results/centralized').mkdir(parents=True, exist_ok=True)
save_evaluation_report(
    y_test, y_pred, y_pred_proba,
    model_name="RandomForest",
    save_dir='../../results/centralized'
)

print("\n" + "="*70)
print("✅ RANDOM FOREST COMPLETE")
print("="*70)
print(f"\nAccuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"AUC-ROC: {metrics.get('roc_auc', 0):.4f}")
