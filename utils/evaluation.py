"""
Evaluation Metrics Utilities
Comprehensive evaluation for credit risk models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
VIZ_DIR = PROJECT_ROOT / 'visualization'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities (optional)
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC-ROC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in a formatted table
    
    Args:
        metrics (dict): Metrics dictionary
        model_name (str): Name of the model
    """
    print("\n" + "="*60)
    print(f"{model_name.upper()} - EVALUATION METRICS")
    print("="*60)
    
    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    
    if metrics.get('roc_auc') is not None:
        print(f"\n📈 Probability-based Metrics:")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    
    print(f"\n📋 Confusion Matrix Components:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    
    print("="*60 + "\n")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Model name for title
        save_path (str): Path to save figure (optional)
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good Credit (0)', 'Bad Credit (1)'],
                yticklabels=['Good Credit (0)', 'Bad Credit (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
            ax.text(j+0.5, i+0.5, '', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities
        model_name (str): Model name for title
        save_path (str): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: ROC curve figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities
        model_name (str): Model name
        save_path (str): Save path
        
    Returns:
        matplotlib.figure.Figure: PR curve figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to {save_path}")
    
    return fig


def compare_models(results_dict, metric='accuracy', save_path=None):
    """
    Compare multiple models
    
    Args:
        results_dict (dict): Dictionary of {model_name: metrics_dict}
        metric (str): Metric to compare
        save_path (str): Save path
        
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")
    
    return fig


def create_metrics_comparison_table(results_dict):
    """
    Create comprehensive comparison table
    
    Args:
        results_dict (dict): Dictionary of {model_name: metrics_dict}
        
    Returns:
        pd.DataFrame: Comparison table
    """
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    data = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        for metric in metrics_list:
            if metric in metrics and metrics[metric] is not None:
                row[metric.replace('_', ' ').title()] = f"{metrics[metric]:.4f}"
            else:
                row[metric.replace('_', ' ').title()] = 'N/A'
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def save_evaluation_report(y_true, y_pred, y_pred_proba, model_name, save_dir=None):
    """
    Generate and save complete evaluation report
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities
        model_name (str): Model name
        save_dir (str): Directory to save reports
    """
    if save_dir is None:
        save_dir = VIZ_DIR
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print metrics
    print_metrics(metrics, model_name)
    
    # Generate plots
    print(f"\nGenerating evaluation plots for {model_name}...")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, model_name, 
                         save_path=save_dir / f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # ROC curve
    if y_pred_proba is not None:
        plot_roc_curve(y_true, y_pred_proba, model_name,
                      save_path=save_dir / f'{model_name}_roc_curve.png')
        plt.close()
        
        plot_precision_recall_curve(y_true, y_pred_proba, model_name,
                                   save_path=save_dir / f'{model_name}_pr_curve.png')
        plt.close()
    
    # Save metrics to JSON
    import json
    metrics_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                           for k, v in metrics.items()}
    
    with open(save_dir / f'{model_name}_metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"✓ Evaluation report saved to {save_dir}")
    
    return metrics


if __name__ == "__main__":
    """
    Test evaluation utilities
    """
    print("\n" + "="*70)
    print("TESTING EVALUATION UTILITIES")
    print("="*70 + "\n")
    
    # Generate dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_pred_proba = np.random.rand(200)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Test metrics calculation
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print_metrics(metrics, "Test Model")
    
    # Test plotting
    plot_confusion_matrix(y_true, y_pred, "Test Model")
    plot_roc_curve(y_true, y_pred_proba, "Test Model")
    plt.show()
    
    print("\n✅ Evaluation utilities test complete!")
