# Federated Learning Implementation

This directory contains the federated learning framework for distributed credit risk prediction.

## Components (Phase 3)

### 1. Server (`server.py`)
- Global model aggregation
- FedAvg algorithm implementation
- Model distribution to clients
- Convergence monitoring

### 2. Client (`client.py`)
- Local model training
- Gradient computation
- Model weight updates
- Local evaluation

### 3. Coordinator (`coordinator.py`)
- Orchestrates federated rounds
- Manages client communication
- Handles data partitioning
- Tracks experiment metrics

## Federated Learning Workflow

```
1. Server initializes global model
2. Server sends model to all clients
3. Each client trains on local data
4. Clients send updates to server
5. Server aggregates updates (FedAvg)
6. Repeat steps 2-5 for N rounds
7. Final global model evaluation
```

## Key Parameters

- **Number of clients:** 3-5
- **Federated rounds:** 20-50
- **Local epochs:** 3-5 per round
- **Batch size:** 32-64
- **Learning rate:** 0.001-0.01

## Data Distribution Strategies

### IID (Independent and Identically Distributed)
- Random split of data
- Each client has similar distribution
- Easier to converge

### Non-IID 
- Heterogeneous data across clients
- Simulates real-world scenario
- More challenging convergence

## Status

- [ ] Server implementation - Week 5
- [ ] Client implementation - Week 5  
- [ ] Coordinator - Week 5
- [ ] IID experiments - Week 5
- [ ] Non-IID experiments - Week 6

## Expected Results

- FL accuracy within 2-3% of centralized
- Communication efficiency analysis
- Privacy-utility tradeoff demonstration
