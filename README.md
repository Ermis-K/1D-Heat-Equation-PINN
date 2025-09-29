# 1D-Heat-Equation-PINN

A Physics-Informed Neural Network (PINN) implementation for solving the one-dimensional heat equation  over space and time, enforcing physical consistency in predictions.

---

# Repository Structure

| File / Folder        | Purpose / Description |
|-----------------------|------------------------|
| `pinn.py`             | Core PINN architecture |
| `train_2.py`          | Training script (data loading, training loop) |
| `sampling.py`         | Functions to sample collocation points, boundary, and initial-condition points |
| `losses.py`           | Loss definitions: PDE residual loss, boundary loss, initial-condition loss |
| `plotting.py`         | Utility functions to visualize predicted vs true solutions |
| `heat_data.mat`       | Reference / ground-truth data for benchmarking and error evaluation |
| `results_heat.png`    | Example output plot of the predicted vs true solution |
| `adaptive_weigths.py` | Algorithm for dyncamically changing the lamda weights, source: https://arxiv.org/abs/2001.04536 |
---
# Prerequisites

- Python 3.8+  
- Libraries: `numpy`, `torch`, `scipy`, `matplotlib`, `h5py` (for `.mat` file reading)  
- GPU support is optional but recommended for faster training
