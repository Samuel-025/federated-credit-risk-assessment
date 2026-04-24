"""
Federated Learning Training Experiments
Runs IID and Non-IID experiments and compares with centralized models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.federated.coordinator import FederatedCoordinator
from utils.evaluation import calculate_metrics, print_metrics


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load preprocessed data (V3 - MinMax, best for neural networks)"""
    print("Loading preprocessed data (V3 - MinMax scaled)...")

    data_dir = Path('data/processed')

    X_train = np.load(data_dir / 'v3_minmax_smote_X_train.npy')
    y_train = np.load(data_dir / 'v3_minmax_smote_y_train.npy')
    X_test  = np.load(data_dir / 'v3_minmax_smote_X_test.npy')
    y_test  = np.load(data_dir / 'v3_minmax_smote_y_test.npy')

    print(f"✓ Data loaded:")
    print(f"  Train: {X_train.shape} | Classes: {np.bincount(y_train.astype(int))}")
    print(f"  Test:  {X_test.shape}  | Classes: {np.bincount(y_test.astype(int))}")

    return X_train, y_train, X_test, y_test


# ============================================================
# EXPERIMENT 1: IID FEDERATED LEARNING
# ============================================================

def run_iid_experiment(X_train, y_train, X_test, y_test,
                       n_clients=3, n_rounds=20, local_epochs=5):
    """
    Experiment 1: IID Data Distribution
    Each client gets randomly assigned data - ideal scenario
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: IID FEDERATED LEARNING")
    print("=" * 70)
    print("Scenario: Each bank receives randomly distributed data")
    print("This represents the ideal federated learning scenario")

    coordinator = FederatedCoordinator(
        n_clients=n_clients,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        architecture=[64, 32],
        dropout=0.3,
        random_state=42
    )

    # Setup with IID data
    coordinator.setup(X_train, y_train, X_test, y_test, method='iid')

    # Train
    results = coordinator.train(verbose=True)

    # Evaluate per client
    client_metrics = coordinator.evaluate_clients()

    # Save results
    coordinator.save_results('experiment_1_iid')

    return results, client_metrics, coordinator


# ============================================================
# EXPERIMENT 2: NON-IID FEDERATED LEARNING
# ============================================================

def run_non_iid_experiment(X_train, y_train, X_test, y_test,
                           n_clients=3, n_rounds=20, local_epochs=5):
    """
    Experiment 2: Non-IID Data Distribution
    Each client has different class distribution - realistic scenario
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: NON-IID FEDERATED LEARNING")
    print("=" * 70)
    print("Scenario: Each bank has different customer demographics")
    print("Client 1: Mostly good credit customers")
    print("Client 2: Balanced mix")
    print("Client 3: Mostly bad credit customers")
    print("This represents the realistic banking scenario")

    coordinator = FederatedCoordinator(
        n_clients=n_clients,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        architecture=[64, 32],
        dropout=0.3,
        random_state=42
    )

    # Setup with Non-IID data
    coordinator.setup(X_train, y_train, X_test, y_test, method='non_iid')

    # Train
    results = coordinator.train(verbose=True)

    # Evaluate per client
    client_metrics = coordinator.evaluate_clients()

    # Save results
    coordinator.save_results('experiment_2_non_iid')

    return results, client_metrics, coordinator


# ============================================================
# EXPERIMENT 3: VARYING NUMBER OF ROUNDS
# ============================================================

def run_rounds_experiment(X_train, y_train, X_test, y_test):
    """
    Experiment 3: Effect of communication rounds
    Tests convergence at 10, 20, 30, 50 rounds
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: COMMUNICATION ROUNDS ANALYSIS")
    print("=" * 70)

    rounds_to_test = [10, 20, 30, 50]
    rounds_results = {}

    for n_rounds in rounds_to_test:
        print(f"\n▶ Testing {n_rounds} rounds...")

        coordinator = FederatedCoordinator(
            n_clients=3,
            n_rounds=n_rounds,
            local_epochs=5,
            architecture=[64, 32],
            random_state=42
        )
        coordinator.setup(X_train, y_train, X_test, y_test, method='iid')
        results = coordinator.train(verbose=False)

        final = results['final_metrics']
        rounds_results[n_rounds] = {
            'accuracy': final['accuracy'],
            'f1_score': final['f1_score'],
            'roc_auc': final['roc_auc'],
            'time': results['total_time']
        }

        print(f"  ✓ {n_rounds} rounds: Accuracy={final['accuracy']:.4f} | "
              f"Time={results['total_time']:.1f}s")

    return rounds_results


# ============================================================
# FINAL COMPARISON: FL vs CENTRALIZED
# ============================================================

def compare_fl_vs_centralized(iid_results, non_iid_results):
    """
    Final comparison between FL and centralized models
    """
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: CENTRALIZED vs FEDERATED LEARNING")
    print("=" * 70)

    # Centralized results (from Phase 2)
    centralized = {
        'Logistic Regression': {
            'accuracy': 0.7333, 'precision': 0.5500,
            'recall': 0.6111, 'f1_score': 0.5789, 'roc_auc': 0.7611
        },
        'Random Forest': {
            'accuracy': 0.7633, 'precision': 0.6173,
            'recall': 0.5556, 'f1_score': 0.5848, 'roc_auc': 0.7943
        },
        'Neural Network (Centralized)': {
            'accuracy': 0.7767, 'precision': 0.6353,
            'recall': 0.6000, 'f1_score': 0.6171, 'roc_auc': 0.7870
        }
    }

    # FL results
    iid_final = iid_results['final_metrics']
    non_iid_final = non_iid_results['final_metrics']

    fl_results = {
        'FL - IID (3 clients)': {
            'accuracy': iid_final['accuracy'],
            'precision': iid_final['precision'],
            'recall': iid_final['recall'],
            'f1_score': iid_final['f1_score'],
            'roc_auc': iid_final['roc_auc']
        },
        'FL - Non-IID (3 clients)': {
            'accuracy': non_iid_final['accuracy'],
            'precision': non_iid_final['precision'],
            'recall': non_iid_final['recall'],
            'f1_score': non_iid_final['f1_score'],
            'roc_auc': non_iid_final['roc_auc']
        }
    }

    all_results = {**centralized, **fl_results}

    # Create comparison table
    rows = []
    for model, metrics in all_results.items():
        rows.append({
            'Model': model,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'AUC-ROC': f"{metrics['roc_auc']:.4f}"
        })

    comparison_df = pd.DataFrame(rows)

    print("\n" + comparison_df.to_string(index=False))

    # Performance gap analysis
    best_centralized_acc = 0.7767  # Neural Network
    iid_acc = iid_final['accuracy']
    non_iid_acc = non_iid_final['accuracy']

    print(f"\n📊 Performance Gap Analysis:")
    print(f"  Best Centralized (NN): {best_centralized_acc:.4f}")
    print(f"  FL - IID:              {iid_acc:.4f} "
          f"(gap: {abs(best_centralized_acc - iid_acc):.4f})")
    print(f"  FL - Non-IID:          {non_iid_acc:.4f} "
          f"(gap: {abs(best_centralized_acc - non_iid_acc):.4f})")

    within_3pct = iid_acc >= (best_centralized_acc - 0.03)
    print(f"\n  ✓ FL within 3% of centralized: {'YES ✅' if within_3pct else 'NO ❌'}")

    # Save comparison
    results_dir = Path('results/federated')
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir / 'fl_vs_centralized_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to results/federated/fl_vs_centralized_comparison.csv")

    return comparison_df, all_results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_convergence(iid_coordinator, non_iid_coordinator):
    """Plot FL convergence curves for both IID and Non-IID"""

    iid_history = iid_coordinator.server.get_convergence_history()
    non_iid_history = non_iid_coordinator.server.get_convergence_history()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_to_plot = [
        ('accuracy', 'Accuracy'),
        ('f1_score', 'F1-Score'),
        ('roc_auc', 'AUC-ROC')
    ]

    for idx, (metric, label) in enumerate(metrics_to_plot):
        axes[idx].plot(iid_history['rounds'], iid_history[metric],
                      marker='o', linewidth=2, markersize=4,
                      label='FL - IID', color='#3498db')
        axes[idx].plot(non_iid_history['rounds'], non_iid_history[metric],
                      marker='s', linewidth=2, markersize=4,
                      label='FL - Non-IID', color='#e74c3c', linestyle='--')

        # Add centralized benchmarks as horizontal lines
        if metric == 'accuracy':
            axes[idx].axhline(y=0.7767, color='#2ecc71', linestyle=':',
                             linewidth=2, label='Centralized NN (0.7767)')
        elif metric == 'f1_score':
            axes[idx].axhline(y=0.6171, color='#2ecc71', linestyle=':',
                             linewidth=2, label='Centralized NN (0.6171)')
        elif metric == 'roc_auc':
            axes[idx].axhline(y=0.7943, color='#f39c12', linestyle=':',
                             linewidth=2, label='Centralized RF (0.7943)')

        axes[idx].set_xlabel('Federated Round', fontsize=11)
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].set_title(f'FL Convergence - {label}', fontsize=13, fontweight='bold')
        axes[idx].legend(loc='lower right', fontsize=9)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim([0.4, 1.0])

    plt.tight_layout()

    viz_dir = Path('visualization')
    viz_dir.mkdir(parents=True, exist_ok=True)
    save_path = viz_dir / 'fl_convergence.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence plot saved to {save_path}")
    plt.close()


def plot_final_comparison(all_results):
    """Plot final comparison bar chart"""
    models = list(all_results.keys())
    metrics = ['accuracy', 'f1_score', 'roc_auc']
    labels = ['Accuracy', 'F1-Score', 'AUC-ROC']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(metrics))
    n_models = len(models)
    width = 0.15

    for i, (model, results) in enumerate(all_results.items()):
        values = [float(results[m]) for m in metrics]
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width,
                     label=model, color=colors[i % len(colors)])

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Centralized vs Federated Learning - Performance Comparison',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    plt.tight_layout()

    save_path = Path('visualization') / 'fl_vs_centralized.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {save_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 3: FEDERATED LEARNING EXPERIMENTS")
    print("Federated Learning for Credit Risk Assessment")
    print("=" * 70)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Experiment 1: IID
    iid_results, iid_client_metrics, iid_coordinator = run_iid_experiment(
        X_train, y_train, X_test, y_test,
        n_clients=3, n_rounds=20, local_epochs=5
    )

    # Experiment 2: Non-IID
    non_iid_results, non_iid_client_metrics, non_iid_coordinator = run_non_iid_experiment(
        X_train, y_train, X_test, y_test,
        n_clients=3, n_rounds=20, local_epochs=5
    )

    # Experiment 3: Varying rounds
    rounds_results = run_rounds_experiment(X_train, y_train, X_test, y_test)

    # Final comparison
    comparison_df, all_results = compare_fl_vs_centralized(iid_results, non_iid_results)

    # Plots
    print("\n📊 Generating visualizations...")
    plot_convergence(iid_coordinator, non_iid_coordinator)
    plot_final_comparison(all_results)

    print("\n" + "=" * 70)
    print("✅ PHASE 3 COMPLETE!")
    print("=" * 70)
    print("\n📁 Results saved to:")
    print("  results/federated/experiment_1_iid/")
    print("  results/federated/experiment_2_non_iid/")
    print("  results/federated/fl_vs_centralized_comparison.csv")
    print("  visualization/fl_convergence.png")
    print("  visualization/fl_vs_centralized.png")
    print("\n📝 Next: Write Chapter 7 & 8 using these results!")


if __name__ == "__main__":
    main()
