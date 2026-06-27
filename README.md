# Federated-Learning-Algorithms-on-MNIST-Dataset

An exploratory Jupyter notebook that implements and compares several federated learning algorithms (FedAvg, FedOpt, FedProx) on the MNIST dataset and contrasts them with a centralized baseline. This repository is experimental / educational — intended to demonstrate algorithmic behaviour under label-heterogeneous (non‑IID) partitions rather than production-ready code or deployments.

## Notebook
- FL-pytorch.ipynb — single notebook containing the full experiment pipeline:
  - data download / preprocessing (MNIST)
  - centralized baseline training and evaluation
  - synthetic non‑IID partitioning of MNIST across 5 simulated clients
  - federated experiments: FedAvg, FedOpt (simulated), FedProx
  - plots of training loss histories and printed test losses / accuracies

You can view the notebook here:
https://github.com/Aditya2814/Federated-Learning-Algorithms-on-MNIST-Dataset/blob/main/FL-pytorch.ipynb

## Summary of what the work shows
- A compact convolutional classifier is trained centrally and via simple federated simulations.
- The dataset is partitioned deliberately to create strong heterogeneity: each simulated client holds only two digit classes (e.g., client1: {0,1}, client2: {2,3}, ...).
- Under this label-skewed partitioning, FedAvg performance drops compared to the centralized baseline. FedOpt (a simple server-optimizer style simulation) and FedProx (proximal regularization in local objectives) are evaluated and partially recover performance.
- The notebook prints evaluation numbers and draws training-loss plots so you can inspect convergence and comparative behaviour.

## Key numeric results (from the notebook runs)
| Experiment | Test loss | Test accuracy |
|---|---:|---:|
| Centralized baseline | ~0.2195 | ~93.83% |
| FedAvg (5 clients, label-skewed) | ~0.7825 | ~74.94% |
| FedOpt (simulated) | ~0.6654 | ~79.21% |
| FedProx (mu=0.01) | ~0.5489 | ~83.60% |

## High-level interpretation
- The centralized model (trained on all MNIST data) achieves the best results as expected.
- The extreme non‑IID (label-skewed) partition used here strongly harms FedAvg — averaging client parameters is not enough when local objectives differ substantially.
- FedOpt and FedProx, as implemented in the notebook, mitigate some of the degradation; FedProx performs best among the federated variants in this experiment. These outcomes qualitatively match expectations from federated learning literature: heterogeneity breaks naive averaging, and server-side optimizers or proximal corrections can help.

## Model and experiment details (concise)
- Dataset
  - MNIST (torchvision), transforms: ToTensor() and Normalize((0.5,), (0.5,))
- Model (PyTorch nn.Module)
  - Conv2d(1 → 10, kernel=5) → MaxPool2d(2) → ReLU
  - Conv2d(10 → 20, kernel=5) → Dropout2d → MaxPool2d(2) → ReLU
  - Flatten → Linear(320 → 50) → ReLU → Dropout → Linear(50 → 10)
- Training / federated setup
  - Centralized baseline: SGD lr=0.001, batch_size=32, epochs=10 (baseline run in notebook)
  - Federated experiments:
    - num_clients = 5 (each client holds two digit classes)
    - local training: SGD lr=0.001, batch_size=32, local epochs=5 per communication round
    - communication rounds: 15
    - FedAvg aggregation: weighted average by client dataset size
    - FedOpt: simulated by aggregating parameter differences and applying a server-side update (illustrative)
    - FedProx: local objective augmented with (mu/2) * ||w_local − w_server||^2, mu=0.01

## Limitations and important notes
- This project is intentionally experimental:
  - The federated setup is simulated inside a single process — there is no networking, secure aggregation, client dropouts, or real federation infrastructure.
  - FedOpt is implemented as a conceptual simulation (parameter differences treated as pseudo-gradients) — not a production FedOpt implementation.
  - Some notebook choices are exploratory (e.g., artificial gradient perturbations in a FedProx training loop) and should not be treated as standard practice.
  - Hyperparameters and random seeds are not exhaustively tuned; results illustrate trends rather than definitive benchmarks.
- The partitioning is extreme (each client has only two classes) to amplify heterogeneity effects; other partitioning strategies will produce different behaviours.

## What you can learn from the repository
- How to implement a minimal PyTorch classifier and training loop for MNIST.
- How to simulate simple federated workflows in a single notebook (model copying, local training, server aggregation).
- Insight into how non‑IID data affects federated learning and how simple algorithmic choices (server optimizers, proximal regularizers) can mitigate the effect.
- A starting point for deeper experiments (more clients, different heterogeneity models, alternative aggregators, reproducibility).

## Possible follow-ups / extensions
- Add a more realistic federated infrastructure (multiple processes or frameworks such as Flower, PySyft or TensorFlow Federated).
- Implement proper server optimizers (FedAdam, FedYogi) and compare rigorously.
- Evaluate additional heterogeneity types (quantity skew, feature shift) and run multiple seeds to report confidence intervals.
- Add experiment logging, checkpoints, and reproducibility (deterministic transforms and fixed seeds).
- Introduce client sampling, compression, privacy (differential privacy), and secure aggregation.

## Contact / author
- Repository: Aditya2814/Federated-Learning-Algorithms-on-MNIST-Dataset
- If you have questions or want to discuss experiments, open an issue in the repository.

## License
- No license file is included in this repository. If you intend for reuse, add a license (MIT, Apache-2.0, etc.).
