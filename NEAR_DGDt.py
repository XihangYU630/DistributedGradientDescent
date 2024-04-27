import numpy as np
import matplotlib.pyplot as plt
from utils import distributed_quadratic_gradient

def near_dgdt(x0, alpha, W, t, A, b, x_optimal, max_iterations):
    """
    NEAR Distributed Gradient Descent with t consensus steps (NEAR-DGDt).
    """

    x = x0
    n = W.shape[0]
    p = A[0].shape[0]
    I_p = np.eye(p)
    Z_t = np.kron(np.linalg.matrix_power(W, t), I_p)

    optimization_error_list = []
    consensus_error_list = []

    for _ in range(max_iterations):

        ## gradient step
        gradient = distributed_quadratic_gradient(x, A, b)
        y = x - alpha * gradient

        # consensus step
        x = Z_t @ y

        x_avg = np.mean(x.reshape(n, p), axis=0)
        x_bar =np.tile(x_avg, n)
        consensus_error = np.linalg.norm(x - x_bar)
        consensus_error_list.append(consensus_error)
        optimization_error = np.linalg.norm(x_avg - x_optimal)**2  # Compute distance to x_optimal
        optimization_error_list.append(optimization_error)

    return x, consensus_error_list, optimization_error_list
