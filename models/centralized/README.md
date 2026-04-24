# Centralized Baseline Models

This directory contains implementations of traditional centralized machine learning models for credit risk prediction.

## Models to be Implemented (Phase 2)

### 1. Logistic Regression (`logistic.py`)
- Simple baseline model
- Good interpretability
- Fast training
- Expected accuracy: ~72-75%

### 2. Random Forest (`random_forest.py`)
- Ensemble method
- Handles non-linear relationships
- Feature importance analysis
- Expected accuracy: ~75-78%

### 3. Neural Network (`neural_net.py`)
- Deep learning approach
- Can capture complex patterns
- Benchmark for federated learning
- Expected accuracy: ~76-80%

## Usage Pattern

```python
from models.centralized import LogisticRegressionModel

# Initialize model
model = LogisticRegressionModel()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)
```

## Status

- [ ] Logistic Regression - To be implemented in Week 3
- [ ] Random Forest - To be implemented in Week 3
- [ ] Neural Network - To be implemented in Week 4

## Notes

All models will use the preprocessed data from `data/processed/` and save results to `results/centralized/`.
