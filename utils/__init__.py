"""
Utility functions for Federated Learning Credit Risk Project
"""

from .data_loader import (
    download_german_credit_data,
    load_german_credit_data,
    get_feature_descriptions,
    print_dataset_info
)

from .preprocessing import (
    CreditDataPreprocessor,
    save_processed_data
)

from .evaluation import (
    calculate_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    compare_models,
    create_metrics_comparison_table,
    save_evaluation_report
)

from .visualization import (
    plot_training_history,
    plot_feature_importance,
    plot_federated_convergence,
    plot_client_performance,
    plot_centralized_vs_federated,
    plot_data_distribution,
    create_results_dashboard
)

__all__ = [
    # Data loading
    'download_german_credit_data',
    'load_german_credit_data',
    'get_feature_descriptions',
    'print_dataset_info',
    
    # Preprocessing
    'CreditDataPreprocessor',
    'save_processed_data',
    
    # Evaluation
    'calculate_metrics',
    'print_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'compare_models',
    'create_metrics_comparison_table',
    'save_evaluation_report',
    
    # Visualization
    'plot_training_history',
    'plot_feature_importance',
    'plot_federated_convergence',
    'plot_client_performance',
    'plot_centralized_vs_federated',
    'plot_data_distribution',
    'create_results_dashboard'
]
