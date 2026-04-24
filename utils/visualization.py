"""
Visualization Utilities
Custom plotting functions for credit risk analysis and federated learning
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VIZ_DIR = PROJECT_ROOT / 'visualization'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')


def plot_training_history(history, metrics=['loss', 'accuracy'], model_name="Model", save_path=None):
    """
    Plot training history for neural networks
    
    Args:
        history: Training history object
        metrics (list): Metrics to plot
        model_name (str): Model name
        save_path (str): Save path
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label=f'Train {metric}', linewidth=2)
            if f'val_{metric}' in history.history:
                axes[idx].plot(history.history[f'val_{metric}'], 
                             label=f'Val {metric}', linewidth=2, linestyle='--')
            
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
            axes[idx].set_title(f'{metric.capitalize()} - {model_name}', 
                              fontsize=12, fontweight='bold')
            axes[idx].legend(loc='best')
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
    
    return fig


def plot_feature_importance(importance_values, feature_names, top_n=20, 
                           model_name="Model", save_path=None):
    """
    Plot feature importance
    
    Args:
        importance_values (array): Feature importance scores
        feature_names (list): Feature names
        top_n (int): Number of top features to display
        model_name (str): Model name
        save_path (str): Save path
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color='steelblue')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance saved to {save_path}")
    
    return fig


def plot_federated_convergence(metrics_per_round, metric_name='accuracy', save_path=None):
    """
    Plot federated learning convergence
    
    Args:
        metrics_per_round (dict): {round: metric_value}
        metric_name (str): Name of metric
        save_path (str): Save path
    """
    rounds = list(metrics_per_round.keys())
    values = list(metrics_per_round.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, values, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax.fill_between(rounds, values, alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('Federated Round', fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(f'Federated Learning Convergence - {metric_name.capitalize()}', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add annotations for first and last
    ax.annotate(f'Start: {values[0]:.4f}', 
               xy=(rounds[0], values[0]), xytext=(rounds[0]+1, values[0]-0.02),
               arrowprops=dict(arrowstyle='->', color='black'),
               fontsize=10, fontweight='bold')
    ax.annotate(f'Final: {values[-1]:.4f}', 
               xy=(rounds[-1], values[-1]), xytext=(rounds[-1]-2, values[-1]+0.02),
               arrowprops=dict(arrowstyle='->', color='black'),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Convergence plot saved to {save_path}")
    
    return fig


def plot_client_performance(client_metrics, save_path=None):
    """
    Plot performance across federated clients
    
    Args:
        client_metrics (dict): {client_id: metrics_dict}
        save_path (str): Save path
    """
    clients = list(client_metrics.keys())
    metrics_to_plot = ['accuracy', 'f1_score', 'roc_auc']
    
    # Prepare data
    data = {metric: [] for metric in metrics_to_plot}
    for client in clients:
        for metric in metrics_to_plot:
            if metric in client_metrics[client]:
                data[metric].append(client_metrics[client][metric])
            else:
                data[metric].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(clients))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for idx, metric in enumerate(metrics_to_plot):
        ax.bar(x + idx*width, data[metric], width, 
               label=metric.replace('_', ' ').title(), color=colors[idx])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_title('Performance Across Federated Clients', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(clients)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Client performance plot saved to {save_path}")
    
    return fig


def plot_centralized_vs_federated(centralized_metrics, federated_metrics, save_path=None):
    """
    Compare centralized vs federated performance
    
    Args:
        centralized_metrics (dict): Centralized model metrics
        federated_metrics (dict): Federated model metrics
        save_path (str): Save path
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    centralized_values = [centralized_metrics.get(m, 0) for m in metrics]
    federated_values = [federated_metrics.get(m, 0) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, centralized_values, width, 
                   label='Centralized', color='#3498db')
    bars2 = ax.bar(x + width/2, federated_values, width, 
                   label='Federated', color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Centralized vs Federated Learning Performance', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")
    
    return fig


def plot_data_distribution(data_splits, save_path=None):
    """
    Plot data distribution across federated clients
    
    Args:
        data_splits (dict): {client_id: (n_samples, class_distribution)}
        save_path (str): Save path
    """
    clients = list(data_splits.keys())
    n_samples = [data_splits[c][0] for c in clients]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample distribution
    ax1.bar(clients, n_samples, color='steelblue')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Client', fontsize=12)
    ax1.set_title('Sample Distribution Across Clients', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(n_samples):
        ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Class distribution
    class_0 = [data_splits[c][1].get(0, 0) for c in clients]
    class_1 = [data_splits[c][1].get(1, 0) for c in clients]
    
    x = np.arange(len(clients))
    width = 0.35
    
    ax2.bar(x - width/2, class_0, width, label='Good Credit (0)', color='#2ecc71')
    ax2.bar(x + width/2, class_1, width, label='Bad Credit (1)', color='#e74c3c')
    
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_xlabel('Client', fontsize=12)
    ax2.set_title('Class Distribution Across Clients', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clients)
    ax2.legend(loc='best')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Data distribution plot saved to {save_path}")
    
    return fig


def create_results_dashboard(all_results, save_path=None):
    """
    Create comprehensive results dashboard
    
    Args:
        all_results (dict): Dictionary with all experiment results
        save_path (str): Save path
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # This is a placeholder - customize based on your needs
    # You can add multiple subplots showing different aspects
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.5, 'Results Dashboard\n(Customize based on experiments)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    """
    Test visualization utilities
    """
    print("\n" + "="*70)
    print("TESTING VISUALIZATION UTILITIES")
    print("="*70 + "\n")
    
    # Test federated convergence plot
    metrics_per_round = {i: 0.6 + 0.3 * (1 - np.exp(-i/5)) for i in range(1, 21)}
    plot_federated_convergence(metrics_per_round, 'accuracy')
    plt.show()
    
    # Test client performance
    client_metrics = {
        'Client 1': {'accuracy': 0.85, 'f1_score': 0.82, 'roc_auc': 0.88},
        'Client 2': {'accuracy': 0.87, 'f1_score': 0.84, 'roc_auc': 0.89},
        'Client 3': {'accuracy': 0.83, 'f1_score': 0.80, 'roc_auc': 0.86},
    }
    plot_client_performance(client_metrics)
    plt.show()
    
    print("\n✅ Visualization utilities test complete!")
