# 1D-Heat-Equation-PINN

A PyTorch implementation of a Physics Informed Neural Network (PINN) designed to solve the  1D heat equation. The solver utilizes a coordinate based Neural Network to approximate the solution $T(x,t)$ by minimizing the residual of the governing equation, initial and boundary condition.

## Project Structure

* **`train_2.py`**: The primary executable. It manages the configuration, and the training loops.
* **`pinn.py`**: Defines the architecture. It employs a standard MLP with `weight_norm` and an input normalization layer to standardize $(x, t)$ inputs to zero mean and unit variance.
* **`sampling.py`**: Handles LHS for the collocation points, initial conditions, and boundary conditions.
* **`losses.py`**: Computes the loss components:
    * $L_{PDE}$
    * $L_{IC}$
    * $L_{BC}$
* **`plotting.py`**: Visualization utilities.

## Dependencies

* `torch` (PyTorch)
* `numpy`
* `scipy`
* `matplotlib`

## Configuration

Parameters are defined in the `config` dictionary. For example:

* Depth, width, and activation function.
*  Learning rate and number of epochs.

## Usage

1.  **Data Requirement:** Requires a reference solution file named `heat_data.mat` .
2.  **Execution:** Run the training script directly:

```bash
python train_2.py
