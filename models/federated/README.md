# Federated Learning Implementation

This directory contains the federated learning framework for privacy-preserving distributed credit risk prediction.

## Implemented Components (Phase 3)

### 1. Server (`server.py`)
- Maintains the global neural network model
- Implements the Federated Averaging (**FedAvg**) aggregation strategy
- Distributes updated global weights to participating client nodes
- Monitors training loss and convergence metrics across rounds

### 2. Client (`client.py`)
- Simulates individual financial institution / client nodes
- Performs local model training on private data partitions
- Computes weight updates and sends local model parameters back to the server
- Evaluates local model performance on held-out test data

### 3. Coordinator (`coordinator.py`)
- Orchestrates multi-round federated training workflows
- Manages client-server communication loops
- Handles IID and Non-IID client data partitioning
- Logs convergence history and final experiment metrics

## Federated Learning Workflow

```
1. Server initializes global model weights
2. Server broadcasts global model to client nodes
3. Each client trains locally on its private data partition
4. Clients upload model updates to the coordinator/server
5. Server aggregates updates using FedAvg
6. Process repeats for N communication rounds
7. Final global model evaluated on central test dataset
```

## Key Configuration Parameters

- **Number of Clients**: 10 client nodes
- **Federated Rounds**: 15-20 rounds
- **Local Epochs**: 3-5 per round
- **Batch Size**: 32
- **Aggregation Algorithm**: Federated Averaging (FedAvg)

## Data Distribution Strategies

### IID (Independent and Identically Distributed)
- Uniform random split across 10 clients
- Equal class distribution across client nodes
- Rapid convergence within 15 rounds

### Non-IID (Heterogeneous Partitioning)
- Skewed feature/class distributions per client simulating real-world banking silos
- Validates model stability under client data heterogeneity

## Status

- [x] Server implementation (`server.py`) - Completed
- [x] Client implementation (`client.py`) - Completed
- [x] Coordinator implementation (`coordinator.py`) - Completed
- [x] Federated Training experiment script (`experiments/federated_training.py`) - Completed
- [x] IID Experiments - Completed (`results/federated/experiment_1_iid/`)
- [x] Non-IID Experiments - Completed (`results/federated/experiment_2_non_iid/`)

## Key Findings

- Global FL model converges within **15 communication rounds**.
- Achieves accuracy within **2%** of the centralized Neural Network while preserving local data privacy.
