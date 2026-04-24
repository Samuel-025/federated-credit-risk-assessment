"""
Federated Learning - Server
Implements FedAvg algorithm for global model aggregation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class FederatedServer:
    """
    Central server in Federated Learning.

    Responsibilities:
    - Maintains the global model
    - Distributes global weights to clients
    - Aggregates client updates using FedAvg
    - Evaluates global model performance
    - Tracks convergence across rounds

    NOTE: Server NEVER sees raw client data.
          It only receives and averages model weights.
    """

    def __init__(self, input_dim, architecture=[64, 32], dropout=0.3):
        """
        Initialize server with global model

        Args:
            input_dim (int): Number of input features
            architecture (list): Hidden layer sizes
            dropout (float): Dropout rate
        """
        self.input_dim = input_dim
        self.architecture = architecture
        self.dropout = dropout
        self.global_model = None
        self.round_metrics = []      # Metrics per FL round
        self.round_loss = []         # Loss per round

        # Build global model
        self._build_global_model()
        print(f"✓ Server initialized | Input dim: {input_dim} | "
              f"Architecture: {architecture}")

    def _build_global_model(self):
        """Build the global neural network model"""
        model = keras.Sequential()

        model.add(keras.layers.Dense(self.architecture[0],
                                     activation='relu',
                                     input_dim=self.input_dim))
        model.add(keras.layers.Dropout(self.dropout))

        for units in self.architecture[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(self.dropout))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.global_model = model

    def get_global_weights(self):
        """
        Get current global model weights to distribute to clients

        Returns:
            list: Global model weights
        """
        return self.global_model.get_weights()

    def federated_averaging(self, client_weights, client_sizes):
        """
        FedAvg Algorithm (McMahan et al., 2017)

        Aggregates client model weights using weighted average.
        Clients with MORE data have MORE influence on global model.

        Formula:
            w_global = Σ (n_k / n_total) * w_k
            where:
                w_k    = weights from client k
                n_k    = number of samples at client k
                n_total = total samples across all clients

        Args:
            client_weights (list): List of weight arrays from each client
            client_sizes (list): Number of training samples per client

        Returns:
            list: Aggregated global weights
        """
        # Calculate total samples
        total_samples = sum(client_sizes)

        # Calculate weighted contribution of each client
        client_proportions = [n / total_samples for n in client_sizes]

        # Initialize aggregated weights
        aggregated_weights = []

        # Iterate over each layer
        for layer_idx in range(len(client_weights[0])):
            # Weighted sum of this layer's weights from all clients
            layer_weighted_avg = np.zeros_like(client_weights[0][layer_idx],
                                               dtype=np.float64)

            for client_idx, (weights, proportion) in enumerate(
                    zip(client_weights, client_proportions)):
                layer_weighted_avg += proportion * weights[layer_idx]

            aggregated_weights.append(layer_weighted_avg)

        return aggregated_weights

    def update_global_model(self, aggregated_weights):
        """
        Update global model with aggregated weights

        Args:
            aggregated_weights (list): Aggregated weights from FedAvg
        """
        self.global_model.set_weights(aggregated_weights)

    def evaluate_global_model(self, X_test, y_test, round_num=None):
        """
        Evaluate global model on held-out test set

        Args:
            X_test (array): Test features
            y_test (array): Test labels
            round_num (int): Current FL round number

        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import (accuracy_score, f1_score,
                                     roc_auc_score, precision_score,
                                     recall_score)

        y_pred_proba = self.global_model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'round': round_num,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }

        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0

        # Store for convergence tracking
        self.round_metrics.append(metrics)

        return metrics

    def save_global_model(self, filepath):
        """
        Save the final global model

        Args:
            filepath (str): Path to save model
        """
        self.global_model.save(filepath)
        print(f"✓ Global model saved to {filepath}")

    def get_convergence_history(self):
        """
        Get metrics history across all rounds

        Returns:
            dict: Metric values per round
        """
        if not self.round_metrics:
            return {}

        history = {
            'rounds': [m['round'] for m in self.round_metrics],
            'accuracy': [m['accuracy'] for m in self.round_metrics],
            'f1_score': [m['f1_score'] for m in self.round_metrics],
            'roc_auc': [m['roc_auc'] for m in self.round_metrics],
            'precision': [m['precision'] for m in self.round_metrics],
            'recall': [m['recall'] for m in self.round_metrics],
        }

        return history
