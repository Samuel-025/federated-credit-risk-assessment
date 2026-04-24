"""
Neural Network - Baseline Model
Copy and run this file to train Neural Network
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append('../..')

from utils.evaluation import calculate_metrics, print_metrics, save_evaluation_report

print("\n" + "="*70)
print("TRAINING NEURAL NETWORK")
print("="*70)

# 1. Load preprocessed data (V3 - MinMax scaled for NN)
print("\n1. Loading data...")
X_train = np.load('../../data/processed/v3_minmax_smote_X_train.npy')
y_train = np.load('../../data/processed/v3_minmax_smote_y_train.npy')
X_test = np.load('../../data/processed/v3_minmax_smote_X_test.npy')
y_test = np.load('../../data/processed/v3_minmax_smote_y_test.npy')

print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")

# 2. Build neural network architecture
print("\n2. Building Neural Network...")
model = keras.Sequential([
    # Input layer + First hidden layer
    keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
    keras.layers.Dropout(0.3),
    
    # Second hidden layer
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    
    # Output layer
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("✓ Model built")
print("\nModel Architecture:")
model.summary()

# 3. Train model
print("\n3. Training Neural Network...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

print("✓ Training complete")

# 4. Evaluate on test set
print("\n4. Evaluating on test set...")
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
print_metrics(metrics, "Neural Network")

# 5. Save model
print("\n5. Saving model...")
Path('../../models/centralized').mkdir(parents=True, exist_ok=True)
model.save('../../models/centralized/neural_network.h5')
print("✓ Model saved to models/centralized/neural_network.h5")

# 6. Save training history
import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv('../../results/centralized/nn_training_history.csv', index=False)

# 7. Save evaluation report
Path('../../results/centralized').mkdir(parents=True, exist_ok=True)
save_evaluation_report(
    y_test, y_pred, y_pred_proba,
    model_name="NeuralNetwork",
    save_dir='../../results/centralized'
)

print("\n" + "="*70)
print("✅ NEURAL NETWORK COMPLETE")
print("="*70)
print(f"\nAccuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"AUC-ROC: {metrics.get('roc_auc', 0):.4f}")
