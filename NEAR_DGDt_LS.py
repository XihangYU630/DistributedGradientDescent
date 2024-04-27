import numpy as np
import matplotlib.pyplot as plt
from utils import distributed_quadratic_gradient, evaluate_quadratic_function

def near_dgdt_ls(x0, alpha, W, t, A, b, x_optimal, max_iterations, alpha_bar=1, tau=0.5, c1=1e-4):
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

        d = -gradient

        alpha = alpha_bar
        f = evaluate_quadratic_function(x, A, b)
        while evaluate_quadratic_function(x+alpha*d, A, b) > f + c1*alpha*gradient.reshape(n*p,1).T @ d.reshape(n*p,1):
            alpha = tau * alpha

        y = x + alpha * d

        # consensus step
        x = Z_t @ y

        x_avg = np.mean(x.reshape(n, p), axis=0)
        x_bar =np.tile(x_avg, n)
        consensus_error = np.linalg.norm(x - x_bar)
        consensus_error_list.append(consensus_error)
        optimization_error = np.linalg.norm(x_avg - x_optimal)**2  # Compute distance to x_optimal
        optimization_error_list.append(optimization_error)

    return x, consensus_error_list, optimization_error_list
