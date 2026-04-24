"""Compare all baseline models"""
import pandas as pd
import json
from pathlib import Path

# Load all metrics
results_dir = Path('../results/centralized')

# Read metrics from JSON files
with open(results_dir / 'LogisticRegression_metrics.json') as f:
    lr_metrics = json.load(f)

with open(results_dir / 'RandomForest_metrics.json') as f:
    rf_metrics = json.load(f)

with open(results_dir / 'NeuralNetwork_metrics.json') as f:
    nn_metrics = json.load(f)

# Create comparison table
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Neural Network'],
    'Accuracy': [lr_metrics['accuracy'], rf_metrics['accuracy'], nn_metrics['accuracy']],
    'Precision': [lr_metrics['precision'], rf_metrics['precision'], nn_metrics['precision']],
    'Recall': [lr_metrics['recall'], rf_metrics['recall'], nn_metrics['recall']],
    'F1-Score': [lr_metrics['f1_score'], rf_metrics['f1_score'], nn_metrics['f1_score']],
    'AUC-ROC': [lr_metrics['roc_auc'], rf_metrics['roc_auc'], nn_metrics['roc_auc']]
})

print("\n" + "="*70)
print("BASELINE MODEL COMPARISON")
print("="*70)
print("\n" + comparison.to_string(index=False))

# Save
comparison.to_csv(results_dir / 'model_comparison.csv', index=False)
print("\n✓ Comparison saved to results/centralized/model_comparison.csv")
