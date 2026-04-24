"""
Federated Learning - Client
Handles local training at each simulated bank/institution
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import copy


class FederatedClient:
    """
    Represents one federated learning client (simulated bank/institution).
    
    Each client:
    - Has its own local dataset (not shared with anyone)
    - Trains a local copy of the global model
    - Sends ONLY model weights back to server (never raw data)
    """

    def __init__(self, client_id, X_train, y_train, X_test, y_test):
        """
        Initialize a federated client

        Args:
            client_id (str): Unique identifier e.g. 'client_1'
            X_train (array): Local training features
            y_train (array): Local training labels
            X_test (array): Local test features
            y_test (array): Local test labels
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.history = []

        print(f"  ✓ {client_id} initialized | "
              f"Train: {len(X_train)} samples | "
              f"Test: {len(X_test)} samples | "
              f"Class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    def build_model(self, input_dim, architecture=[64, 32], dropout=0.3):
        """
        Build local neural network (same architecture as centralized)

        Args:
            input_dim (int): Number of input features
            architecture (list): Hidden layer sizes
            dropout (float): Dropout rate
        """
        model = keras.Sequential()

        # First hidden layer
        model.add(keras.layers.Dense(architecture[0],
                                     activation='relu',
                                     input_dim=input_dim))
        model.add(keras.layers.Dropout(dropout))

        # Additional hidden layers
        for units in architecture[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(dropout))

        # Output layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return self

    def set_weights(self, global_weights):
        """
        Receive global model weights from server and update local model

        Args:
            global_weights (list): Global model weights from server
        """
        self.model.set_weights(global_weights)

    def get_weights(self):
        """
        Return local model weights to send back to server
        NOTE: Only weights are shared, NEVER raw data

        Returns:
            list: Current model weights
        """
        return self.model.get_weights()

    def local_train(self, epochs=5, batch_size=32, verbose=0):
        """
        Train model on local data for specified number of epochs

        Args:
            epochs (int): Local training epochs per FL round
            batch_size (int): Batch size
            verbose (int): Verbosity level

        Returns:
            dict: Training history (loss, accuracy)
        """
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.1
        )

        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]

        self.history.append({
            'loss': train_loss,
            'accuracy': train_acc
        })

        return {'loss': train_loss, 'accuracy': train_acc}

    def evaluate(self):
        """
        Evaluate model on local test data

        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import (accuracy_score, f1_score,
                                     roc_auc_score, precision_score,
                                     recall_score)

        y_pred_proba = self.model.predict(self.X_test, verbose=0).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'client_id': self.client_id,
            'n_samples': len(self.X_train),
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
        }

        try:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0

        return metrics

    def get_sample_count(self):
        """Return number of local training samples (used for weighted aggregation)"""
        return len(self.X_train)
