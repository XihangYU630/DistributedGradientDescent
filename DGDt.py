import numpy as np
import matplotlib.pyplot as plt
from utils import distributed_quadratic_gradient

def dgd_t(x0, alpha, W, t, A, b, x_optimal, max_iterations):
    """
    Distributed Gradient Descent with t consensus steps (DGDt) for quadratic functions.
    This version tracks the convergence towards the optimal solution x_optimal.

    Parameters:
    - x0: Initial point (numpy array of shape (n*p,)).
    - alpha: Step size (float).
    - W: Consensus weight matrix (numpy matrix of shape (n, n)).
    - t: Number of consensus steps (int).
    - A: List of matrices A_i for each node (list of numpy matrices).
    - b: List of vectors b_i for each node (list of numpy arrays).
    - x_optimal: The optimal solution computed centrally.
    - max_iterations: Maximum number of iterations (int).

    Returns:
    - x: The final iterate after convergence or maximum iterations (numpy array of shape (n*p,)).
    - distances: List of Euclidean distances from the current iterate to the optimal solution.
    """
    x = x0
    n = W.shape[0]
    p = A[0].shape[0]
    I_p = np.eye(p)  # Identity matrix of size p
    Z_t = np.kron(np.linalg.matrix_power(W, t), I_p)  # Compute Z_t as W^t ⊗ I_p
    optimization_error_list = []
    consensus_error_list = []

    for _ in range(max_iterations):
        gradient = distributed_quadratic_gradient(x, A, b)
        x = Z_t @ x - alpha * gradient
        x_avg = np.mean(x.reshape(n, p), axis=0)  # Calculate the average of the vectors
        x_bar =np.tile(x_avg, n)
        consensus_error = np.linalg.norm(x - x_bar)
        consensus_error_list.append(consensus_error)
        optimization_error = np.linalg.norm(x_avg - x_optimal)**2  # Compute distance to x_optimal
        optimization_error_list.append(optimization_error)

    return x, consensus_error_list, optimization_error_list
