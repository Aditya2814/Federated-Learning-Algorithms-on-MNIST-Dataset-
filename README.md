# Federated Learning Algorithms on MNIST Dataset

A comprehensive implementation and exploration of federated learning algorithms applied to the MNIST handwritten digit classification dataset.

## Overview

This project demonstrates the core concepts and practical implementation of federated learning - a machine learning approach that enables training models across decentralized data sources without centralizing raw data. The implementation uses the MNIST dataset as a benchmark for exploring various federated learning techniques and algorithms.

## Project Focus

### Federated Learning Concepts

This work explores:

- **Decentralized Training**: Implementing distributed machine learning where multiple clients train models locally on their private data
- **Model Aggregation**: Techniques for combining locally-trained models into a global model
- **Privacy-Preserving Machine Learning**: Ensuring sensitive data never leaves the client devices during training
- **Communication Efficiency**: Strategies to reduce communication overhead in federated settings

### Algorithms Implemented

The project implements and compares various federated learning algorithms including:

- **Federated Averaging (FedAvg)**: The foundational algorithm for federated learning, averaging locally updated model parameters
- **PyTorch-based Implementation**: Leveraging PyTorch for efficient model training and parameter manipulation

### Dataset

- **MNIST**: 70,000 handwritten digit images (0-9)
- **Distributed Simulation**: Dataset is partitioned across simulated clients to simulate a federated environment
- **Non-IID Data**: Explores scenarios with non-independent and identically distributed (non-IID) data across clients, reflecting real-world federated settings

## Key Experiments

The notebook explores:

1. **Centralized vs. Federated Training**: Comparing traditional centralized learning with federated approaches
2. **Impact of Number of Clients**: Understanding how scaling the number of participating clients affects model convergence and accuracy
3. **Local Epochs**: Investigating the trade-off between local training iterations and communication rounds
4. **Data Distribution**: Analyzing performance under various data heterogeneity scenarios
5. **Model Convergence**: Tracking how federated models converge compared to centralized baselines

## Architecture

```
Clients (Local Training)
    ↓
Local Model Updates
    ↓
Central Server (Aggregation)
    ↓
Global Model Distribution
    ↓
Repeat
```

Each client:
- Maintains local data (MNIST subset)
- Trains model locally for E epochs
- Sends model updates to the server

Server:
- Aggregates updates from participating clients
- Computes weighted average of model parameters
- Distributes updated global model back to clients

## Technical Implementation

- **Framework**: PyTorch for neural network implementation
- **Language**: Python 3
- **Notebook Format**: Jupyter Notebook for interactive exploration and visualization

## Results & Insights

The experiments provide insights into:

- How federated learning converges compared to centralized training
- The effects of data heterogeneity on model performance
- Communication vs. computation trade-offs in federated settings
- Practical considerations for deploying federated learning systems

## Research Significance

This project demonstrates that:

1. **Federated learning is feasible**: Models can achieve competitive accuracy without centralizing data
2. **Communication is key**: The number and size of communication rounds is a critical bottleneck
3. **Non-IID data is challenging**: Data heterogeneity across clients significantly impacts convergence
4. **Practical trade-offs exist**: There are multiple ways to balance accuracy, communication, and privacy

## References & Background

Federated learning builds on decades of distributed machine learning research while introducing new privacy considerations. This implementation is inspired by foundational work in the field, particularly the FedAvg algorithm from "Federated Learning: Communication-Efficient Learning of Deep Networks from Decentralized Data".

## Purpose

This is a research and educational project designed to:

- Understand federated learning fundamentals
- Explore algorithm implementations and variations
- Analyze performance characteristics in controlled settings
- Serve as a reference for federated learning concepts

---

**Created**: July 2024  
**Author**: Aditya2814
