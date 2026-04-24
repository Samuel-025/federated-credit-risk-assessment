"""
Federated Learning - Coordinator
Orchestrates the complete FL training process
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.federated.client import FederatedClient
from models.federated.server import FederatedServer


class FederatedCoordinator:
    """
    Orchestrates the complete federated learning workflow.

    Workflow per round:
        1. Server sends global weights to all clients
        2. Each client trains locally on private data
        3. Each client sends updated weights back to server
        4. Server aggregates weights using FedAvg
        5. Server evaluates updated global model
        6. Repeat for N rounds
    """

    def __init__(self, n_clients=3, n_rounds=20, local_epochs=5,
                 architecture=[64, 32], dropout=0.3, random_state=42):
        """
        Initialize the coordinator

        Args:
            n_clients (int): Number of federated clients (simulated banks)
            n_rounds (int): Number of federated rounds
            local_epochs (int): Local training epochs per round
            architecture (list): Neural network hidden layer sizes
            dropout (float): Dropout rate
            random_state (int): Random seed
        """
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.architecture = architecture
        self.dropout = dropout
        self.random_state = random_state

        self.clients = []
        self.server = None
        self.X_test_global = None
        self.y_test_global = None
        self.results = {}

        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / 'results' / 'federated'
        self.models_dir = self.project_root / 'models' / 'federated'
        self.splits_dir = self.project_root / 'data' / 'federated_splits'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(random_state)

    def create_federated_splits(self, X, y, method='iid'):
        """
        Split data across clients to simulate multiple banks

        Args:
            X (array): Features
            y (array): Labels
            method (str): 'iid' for uniform, 'non_iid' for heterogeneous

        Returns:
            dict: {client_id: (X_train, y_train, X_test, y_test)}
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        client_data = {}

        if method == 'iid':
            # IID: Randomly shuffle and split evenly
            np.random.shuffle(indices)
            client_indices = np.array_split(indices, self.n_clients)

        else:  # non_iid
            # Non-IID: Each client has different class distribution
            # Simulates banks serving different demographics
            class_0_idx = indices[y == 0]
            class_1_idx = indices[y == 1]

            np.random.shuffle(class_0_idx)
            np.random.shuffle(class_1_idx)

            if self.n_clients == 3:
                # Proportions for class 0: 50%, 30%, 20%
                # Proportions for class 1: 20%, 30%, 50%
                c0_cut = [int(len(class_0_idx) * p) for p in [0.50, 0.80]]
                c1_cut = [int(len(class_1_idx) * p) for p in [0.20, 0.50]]

                c0_splits = np.split(class_0_idx, c0_cut)
                c1_splits = np.split(class_1_idx, c1_cut)

                client_indices = []
                for i in range(3):
                    combined = np.concatenate([c0_splits[i], c1_splits[i]])
                    np.random.shuffle(combined)
                    client_indices.append(combined)

        # Create train/test split for each client
        for i, idx in enumerate(client_indices):
            client_id = f'client_{i + 1}'
            X_client = X[idx]
            y_client = y[idx]

            # 80-20 split within each client
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_client, y_client,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_client if len(np.unique(y_client)) > 1 else None
            )
            client_data[client_id] = (X_tr, y_tr, X_te, y_te)

        return client_data

    def setup(self, X_train, y_train, X_test, y_test, method='iid'):
        """
        Set up clients and server

        Args:
            X_train (array): Full training features
            y_train (array): Full training labels
            X_test (array): Global test features
            y_test (array): Global test labels
            method (str): Data split method ('iid' or 'non_iid')
        """
        print("\n" + "=" * 70)
        print(f"FEDERATED LEARNING SETUP ({method.upper()})")
        print("=" * 70)

        self.X_test_global = X_test
        self.y_test_global = y_test

        input_dim = X_train.shape[1]

        # Create federated data splits
        print(f"\n📊 Splitting data across {self.n_clients} clients ({method})...")
        client_data = self.create_federated_splits(X_train, y_train, method)

        # Initialize clients
        print(f"\n🏦 Initializing {self.n_clients} clients (simulated banks):")
        self.clients = []
        for client_id, (X_tr, y_tr, X_te, y_te) in client_data.items():
            client = FederatedClient(client_id, X_tr, y_tr, X_te, y_te)
            client.build_model(input_dim, self.architecture, self.dropout)
            self.clients.append(client)

        # Initialize server
        print(f"\n🖥️  Initializing server...")
        self.server = FederatedServer(input_dim, self.architecture, self.dropout)

        print(f"\n✓ Setup complete!")
        print(f"  - Clients: {self.n_clients}")
        print(f"  - Rounds: {self.n_rounds}")
        print(f"  - Local epochs per round: {self.local_epochs}")
        print(f"  - Model architecture: {input_dim} → {self.architecture} → 1")
        print("=" * 70)

    def train(self, verbose=True):
        """
        Run the complete federated learning training process

        Args:
            verbose (bool): Print progress each round

        Returns:
            dict: Complete training history and final metrics
        """
        print("\n" + "=" * 70)
        print("FEDERATED LEARNING TRAINING - FedAvg Algorithm")
        print("=" * 70)
        print("\nEach round:")
        print("  1. Server → Clients: Distribute global weights")
        print("  2. Clients: Train locally on private data")
        print("  3. Clients → Server: Send updated weights")
        print("  4. Server: Aggregate with FedAvg")
        print("  5. Server: Evaluate global model")
        print("=" * 70 + "\n")

        start_time = time.time()

        for round_num in range(1, self.n_rounds + 1):
            # ──────────────────────────────────────────
            # STEP 1: Server broadcasts global weights
            # ──────────────────────────────────────────
            global_weights = self.server.get_global_weights()

            # ──────────────────────────────────────────
            # STEP 2 & 3: Clients train locally and return weights
            # ──────────────────────────────────────────
            client_weights = []
            client_sizes = []
            client_metrics_this_round = []

            for client in self.clients:
                # Client receives global weights
                client.set_weights(global_weights)

                # Client trains locally
                train_result = client.local_train(
                    epochs=self.local_epochs,
                    batch_size=32,
                    verbose=0
                )

                # Client returns updated weights
                client_weights.append(client.get_weights())
                client_sizes.append(client.get_sample_count())
                client_metrics_this_round.append(train_result)

            # ──────────────────────────────────────────
            # STEP 4: Server aggregates (FedAvg)
            # ──────────────────────────────────────────
            aggregated_weights = self.server.federated_averaging(
                client_weights, client_sizes
            )
            self.server.update_global_model(aggregated_weights)

            # ──────────────────────────────────────────
            # STEP 5: Evaluate global model
            # ──────────────────────────────────────────
            global_metrics = self.server.evaluate_global_model(
                self.X_test_global,
                self.y_test_global,
                round_num=round_num
            )

            # Print progress every 5 rounds
            if verbose and (round_num % 5 == 0 or round_num == 1):
                elapsed = time.time() - start_time
                print(f"  Round {round_num:3d}/{self.n_rounds} | "
                      f"Accuracy: {global_metrics['accuracy']:.4f} | "
                      f"F1: {global_metrics['f1_score']:.4f} | "
                      f"AUC: {global_metrics['roc_auc']:.4f} | "
                      f"Time: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"\n✓ Training complete in {total_time:.1f} seconds")

        # Get final metrics
        final_metrics = self.server.round_metrics[-1]
        convergence_history = self.server.get_convergence_history()

        print(f"\n{'='*70}")
        print("FINAL GLOBAL MODEL PERFORMANCE")
        print(f"{'='*70}")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f} "
              f"({final_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")
        print(f"  F1-Score:  {final_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {final_metrics['roc_auc']:.4f}")
        print(f"{'='*70}\n")

        self.results = {
            'final_metrics': final_metrics,
            'convergence_history': convergence_history,
            'total_time': total_time,
            'n_rounds': self.n_rounds,
            'n_clients': self.n_clients,
            'local_epochs': self.local_epochs
        }

        return self.results

    def evaluate_clients(self):
        """
        Evaluate each client's local model performance

        Returns:
            pd.DataFrame: Per-client metrics
        """
        print("\n📊 Per-Client Performance:")
        print("-" * 60)

        all_metrics = []
        for client in self.clients:
            metrics = client.evaluate()
            all_metrics.append(metrics)
            print(f"  {metrics['client_id']}: "
                  f"Acc={metrics['accuracy']:.4f} | "
                  f"F1={metrics['f1_score']:.4f} | "
                  f"AUC={metrics['roc_auc']:.4f} | "
                  f"N={metrics['n_samples']}")

        return pd.DataFrame(all_metrics)

    def save_results(self, experiment_name):
        """
        Save all experiment results

        Args:
            experiment_name (str): e.g. 'iid_3clients_20rounds'
        """
        save_dir = self.results_dir / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save convergence history
        if self.results.get('convergence_history'):
            history_df = pd.DataFrame(self.results['convergence_history'])
            history_df.to_csv(save_dir / 'convergence_history.csv', index=False)

        # Save final metrics
        if self.results.get('final_metrics'):
            with open(save_dir / 'final_metrics.json', 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                metrics = {k: float(v) if isinstance(v, (np.floating, np.integer))
                          else v for k, v in self.results['final_metrics'].items()}
                json.dump(metrics, f, indent=2)

        # Save global model
        model_path = save_dir / 'global_model.h5'
        self.server.save_global_model(str(model_path))

        print(f"\n✓ Results saved to: {save_dir}")

        return save_dir
