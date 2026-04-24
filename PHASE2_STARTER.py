# ========================================================================
# PHASE 2 STARTER CODE - Copy this entire file and save as phase2_start.py
# ========================================================================

"""
Federated Learning Credit Risk - Phase 2 Quick Start
Run this file to create preprocessed data and train baseline models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========================================================================
# STEP 1: DATA PREPROCESSING
# ========================================================================

def preprocess_data():
    """Create 3 versions of preprocessed data"""
    
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    # Import utilities (make sure utils folder exists in your project)
    from utils.data_loader import load_german_credit_data
    from utils.preprocessing import CreditDataPreprocessor, save_processed_data
    
    # Load data
    print("\n1. Loading German Credit Dataset...")
    df = load_german_credit_data()
    print(f"✓ Data loaded: {df.shape}")
    
    # Version 1: Standard scaling, no balancing (for Random Forest)
    print("\n2. Creating V1: Standard + No Balancing...")
    prep_v1 = CreditDataPreprocessor()
    result_v1 = prep_v1.preprocess_pipeline(
        df, 
        encoding='label', 
        scaling='standard',
        feature_eng=True, 
        balance_method='none'
    )
    save_processed_data(result_v1, prefix='v1_standard_nobalance')
    print("✓ V1 saved")
    
    # Version 2: Standard scaling + SMOTE (for Logistic Regression)
    print("\n3. Creating V2: Standard + SMOTE...")
    prep_v2 = CreditDataPreprocessor()
    result_v2 = prep_v2.preprocess_pipeline(
        df, 
        encoding='label', 
        scaling='standard',
        feature_eng=True, 
        balance_method='smote'
    )
    save_processed_data(result_v2, prefix='v2_standard_smote')
    print("✓ V2 saved")
    
    # Version 3: MinMax scaling + SMOTE (for Neural Network)
    print("\n4. Creating V3: MinMax + SMOTE...")
    prep_v3 = CreditDataPreprocessor()
    result_v3 = prep_v3.preprocess_pipeline(
        df, 
        encoding='label', 
        scaling='minmax',
        feature_eng=True, 
        balance_method='smote'
    )
    save_processed_data(result_v3, prefix='v3_minmax_smote')
    print("✓ V3 saved")
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE - 3 versions created!")
    print("="*70)
    
    return result_v1, result_v2, result_v3


# ========================================================================
# STEP 2: TRAIN LOGISTIC REGRESSION
# ========================================================================

def train_logistic_regression():
    """Train and evaluate Logistic Regression model"""
    
    print("\n" + "="*70)
    print("STEP 2: LOGISTIC REGRESSION")
    print("="*70)
    
    from sklearn.linear_model import LogisticRegression
    from utils.evaluation import calculate_metrics, print_metrics
    
    # Load V2 data (with SMOTE)
    print("\n1. Loading preprocessed data (V2)...")
    X_train = np.load('data/processed/v2_standard_smote_X_train.npy')
    y_train = np.load('data/processed/v2_standard_smote_y_train.npy')
    X_test = np.load('data/processed/v2_standard_smote_X_test.npy')
    y_test = np.load('data/processed/v2_standard_smote_y_test.npy')
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model
    print("\n2. Training Logistic Regression...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    print("\n3. Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Logistic Regression")
    
    # Save model
    import joblib
    Path('models/centralized').mkdir(parents=True, exist_ok=True)
    joblib.dump(model, 'models/centralized/logistic_regression.pkl')
    print("\n✓ Model saved to models/centralized/logistic_regression.pkl")
    
    return model, metrics


# ========================================================================
# STEP 3: TRAIN RANDOM FOREST
# ========================================================================

def train_random_forest():
    """Train and evaluate Random Forest model"""
    
    print("\n" + "="*70)
    print("STEP 3: RANDOM FOREST")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    from utils.evaluation import calculate_metrics, print_metrics
    
    # Load V2 data
    print("\n1. Loading preprocessed data (V2)...")
    X_train = np.load('data/processed/v2_standard_smote_X_train.npy')
    y_train = np.load('data/processed/v2_standard_smote_y_train.npy')
    X_test = np.load('data/processed/v2_standard_smote_X_test.npy')
    y_test = np.load('data/processed/v2_standard_smote_y_test.npy')
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model
    print("\n2. Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    print("\n3. Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Random Forest")
    
    # Feature importance
    print("\n4. Top 10 Important Features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"   {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    # Save model
    import joblib
    joblib.dump(model, 'models/centralized/random_forest.pkl')
    print("\n✓ Model saved to models/centralized/random_forest.pkl")
    
    return model, metrics


# ========================================================================
# STEP 4: TRAIN NEURAL NETWORK
# ========================================================================

def train_neural_network():
    """Train and evaluate Neural Network model"""
    
    print("\n" + "="*70)
    print("STEP 4: NEURAL NETWORK")
    print("="*70)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("\n⚠️  TensorFlow not installed. Install with:")
        print("   pip install tensorflow")
        return None, None
    
    from utils.evaluation import calculate_metrics, print_metrics
    
    # Load V3 data (MinMax scaled)
    print("\n1. Loading preprocessed data (V3 - MinMax)...")
    X_train = np.load('data/processed/v3_minmax_smote_X_train.npy')
    y_train = np.load('data/processed/v3_minmax_smote_y_train.npy')
    X_test = np.load('data/processed/v3_minmax_smote_X_test.npy')
    y_test = np.load('data/processed/v3_minmax_smote_y_test.npy')
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Build model
    print("\n2. Building Neural Network...")
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("✓ Model built")
    print(model.summary())
    
    # Train model
    print("\n3. Training Neural Network...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    print("✓ Training complete")
    
    # Evaluate
    print("\n4. Evaluating model...")
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Neural Network")
    
    # Save model
    model.save('models/centralized/neural_network.h5')
    print("\n✓ Model saved to models/centralized/neural_network.h5")
    
    return model, metrics


# ========================================================================
# STEP 5: COMPARE ALL MODELS
# ========================================================================

def compare_models(lr_metrics, rf_metrics, nn_metrics):
    """Create comparison table and visualizations"""
    
    print("\n" + "="*70)
    print("STEP 5: MODEL COMPARISON")
    print("="*70)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Neural Network'],
        'Accuracy': [
            lr_metrics['accuracy'],
            rf_metrics['accuracy'],
            nn_metrics['accuracy'] if nn_metrics else 0
        ],
        'Precision': [
            lr_metrics['precision'],
            rf_metrics['precision'],
            nn_metrics['precision'] if nn_metrics else 0
        ],
        'Recall': [
            lr_metrics['recall'],
            rf_metrics['recall'],
            nn_metrics['recall'] if nn_metrics else 0
        ],
        'F1-Score': [
            lr_metrics['f1_score'],
            rf_metrics['f1_score'],
            nn_metrics['f1_score'] if nn_metrics else 0
        ],
        'AUC-ROC': [
            lr_metrics.get('roc_auc', 0),
            rf_metrics.get('roc_auc', 0),
            nn_metrics.get('roc_auc', 0) if nn_metrics else 0
        ]
    })
    
    print("\n📊 BASELINE MODEL COMPARISON:")
    print(comparison.to_string(index=False))
    
    # Save table
    Path('results/centralized').mkdir(parents=True, exist_ok=True)
    comparison.to_csv('results/centralized/model_comparison.csv', index=False)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    ax.bar(x - width, comparison.iloc[0, 1:].values, width, label='Logistic Regression')
    ax.bar(x, comparison.iloc[1, 1:].values, width, label='Random Forest')
    if nn_metrics:
        ax.bar(x + width, comparison.iloc[2, 1:].values, width, label='Neural Network')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    Path('visualization').mkdir(parents=True, exist_ok=True)
    plt.savefig('visualization/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved to visualization/baseline_comparison.png")
    
    plt.show()
    
    return comparison


# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    """Run complete Phase 2 pipeline"""
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING CREDIT RISK - PHASE 2")
    print("Data Preprocessing & Baseline Models")
    print("="*70)
    
    try:
        # Step 1: Preprocessing
        print("\n🔄 Starting Step 1: Preprocessing...")
        preprocess_data()
        
        # Step 2: Logistic Regression
        print("\n🔄 Starting Step 2: Logistic Regression...")
        lr_model, lr_metrics = train_logistic_regression()
        
        # Step 3: Random Forest
        print("\n🔄 Starting Step 3: Random Forest...")
        rf_model, rf_metrics = train_random_forest()
        
        # Step 4: Neural Network (optional - requires TensorFlow)
        print("\n🔄 Starting Step 4: Neural Network...")
        nn_model, nn_metrics = train_neural_network()
        
        # Step 5: Compare
        if nn_metrics:
            print("\n🔄 Starting Step 5: Model Comparison...")
            comparison = compare_models(lr_metrics, rf_metrics, nn_metrics)
        else:
            print("\n⚠️  Skipping Neural Network comparison (not trained)")
            comparison = compare_models(lr_metrics, rf_metrics, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0})
        
        print("\n" + "="*70)
        print("✅ PHASE 2 COMPLETE!")
        print("="*70)
        print("\n📁 Files created:")
        print("   - data/processed/v1_*.npy (preprocessing v1)")
        print("   - data/processed/v2_*.npy (preprocessing v2)")
        print("   - data/processed/v3_*.npy (preprocessing v3)")
        print("   - models/centralized/logistic_regression.pkl")
        print("   - models/centralized/random_forest.pkl")
        print("   - models/centralized/neural_network.h5")
        print("   - results/centralized/model_comparison.csv")
        print("   - visualization/baseline_comparison.png")
        
        print("\n📝 Next steps:")
        print("   1. Document results in blackbook (Chapter 6)")
        print("   2. Start Phase 3: Federated Learning")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   1. Make sure utils/ folder exists with all utilities")
        print("   2. Run 'python utils/data_loader.py' first to download data")
        print("   3. Install missing packages: pip install scikit-learn imbalanced-learn")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
