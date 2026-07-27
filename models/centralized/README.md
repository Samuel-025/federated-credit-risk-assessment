# Centralized Baseline Models

This directory contains implementations of traditional centralized machine learning models for credit risk prediction.

## Implemented Models (Phase 2)

### 1. Logistic Regression (`logistic_regression_train.py`)
- Simple baseline linear model
- High interpretability with feature coefficient analysis
- Fast training time (<1s)
- Accuracy achieved: ~73.5%

### 2. Random Forest (`random_forest_train.py`)
- Ensemble method (100 decision trees)
- Captures non-linear feature interactions
- Extracts Gini feature importance rankings
- Accuracy achieved: ~76.8%

### 3. Neural Network (`neural_network_train.py`)
- Deep feedforward neural network (2 hidden layers: 64 & 32 units with ReLU and Dropout)
- Trained using Adam optimizer with Binary Cross-Entropy loss
- Serves as the primary performance benchmark for federated learning
- Accuracy achieved: ~78.2%

## Usage Pattern

```bash
# Run training for centralized models
python models/centralized/logistic_regression_train.py
python models/centralized/random_forest_train.py
python models/centralized/neural_network_train.py

# Run comparative analysis
python experiments/compare_models.py
```

## Status

- [x] Logistic Regression - Completed (`logistic_regression_train.py`)
- [x] Random Forest - Completed (`random_forest_train.py`)
- [x] Neural Network - Completed (`neural_network_train.py`)

## Notes

All models consume preprocessed datasets from `data/processed/` and output metrics and visualizations to `results/centralized/`.
