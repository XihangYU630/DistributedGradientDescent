import numpy as np
import matplotlib.pyplot as plt
from utils import create_4_cyclic_graph, generate_list_of_matrices, compute_global_optimal_solution
from DGDt import dgd_t
from NEAR_DGDt import near_dgdt
from NEAR_DGDt_LS import near_dgdt_ls

np.random.seed(42)

# set Problem
n, p = 10, 10  # Number of nodes and dimension size
x0 = np.random.randn(n * p)  # Example initial point
alpha = 0.1  # Example step size
W = create_4_cyclic_graph(n)  # Random weight matrix for the 4-cyclic graph topology

eigenvalues, _ = np.linalg.eig(W)
print("Eigenvalues of W:", eigenvalues)

t = 2  # Number of consensus steps
max_iterations = 500  # Maximum iterations
condition_number = 10  # Desired condition number


# Define matrices A and vectors b
A = generate_list_of_matrices(n, p, condition_number)
b = [np.random.randn(p) for _ in range(n)]
x_optimal = compute_global_optimal_solution(A, b)

# Cost coefficients
c1, c2 = 1, 1  # Coefficients for communications and gradient evaluations

# set Methods
##############
#### DGDt ####
##############

# Set of different consensus steps to test
t_values = [1, 2, 5, 10]

# Store results for plotting
optimization_error_list_dgdt = []
consensus_error_list_dgdt = []

optimization_error_list_near_dgdt = []
consensus_error_list_near_dgdt = []

optimization_error_list_near_dgdt_ls = []
consensus_error_list_near_dgdt_ls = []


for t in t_values:
    # DGDt
    _, consensus_error_list, optimization_error_list = dgd_t(x0, alpha, W, t, A, b, x_optimal, max_iterations)
    consensus_error_list_dgdt.append(consensus_error_list)
    optimization_error_list_dgdt.append(optimization_error_list)
    
    # NEAR-DGDt
    _, consensus_error_list, optimization_error_list = near_dgdt(x0, alpha, W, t, A, b, x_optimal, max_iterations)
    consensus_error_list_near_dgdt.append(consensus_error_list)
    optimization_error_list_near_dgdt.append(optimization_error_list)

    # NEAR-DGDt-LS
    _, consensus_error_list, optimization_error_list = near_dgdt_ls(x0, alpha, W, t, A, b, x_optimal, max_iterations)
    consensus_error_list_near_dgdt_ls.append(consensus_error_list)
    optimization_error_list_near_dgdt_ls.append(optimization_error_list)

############################
#### Optimization Error ####
############################
    
colors = ['r', '#FFA500', 'c', 'b']

# Plotting all results
plt.figure(figsize=(15, 12))

# Relative Error vs. Number of Iterations
plt.subplot(2, 2, 1)
for index, distances in enumerate(optimization_error_list_dgdt):
    plt.semilogy(distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt):
    plt.semilogy(distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt_ls):
    plt.semilogy(distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])
plt.title('Convergence vs. Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Log of Normalized Distance to Optimal Solution')
plt.grid(True)

# Relative Error vs. Number of Communications
plt.subplot(2, 2, 2)
for index, distances in enumerate(optimization_error_list_dgdt):
    plt.semilogy(np.arange(max_iterations) * t_values[index], distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt):
    plt.semilogy(np.arange(max_iterations) * t_values[index], distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt_ls):
    plt.semilogy(np.arange(max_iterations) * t_values[index], distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])

plt.title('Relative Error vs. Number of Communications')
plt.xlabel('Number of Communications')
plt.ylabel('Log of Normalized Distance to Optimal Solution')
plt.grid(True)

# Relative Error vs. Number of Gradient Evaluations
plt.subplot(2, 2, 3)
for index, distances in enumerate(optimization_error_list_dgdt):
    plt.semilogy(distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt):
    plt.semilogy(distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt_ls):
    plt.semilogy(distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])

plt.title('Relative Error vs. Number of Gradient Evaluations')
plt.xlabel('Number of Gradient Evaluations')
plt.ylabel('Log of Normalized Distance to Optimal Solution')
plt.grid(True)

# Relative Error vs. Cost
plt.subplot(2, 2, 4)
for index, distances in enumerate(optimization_error_list_dgdt):
    cost = c1 * (np.arange(max_iterations) * t_values[index]) + c2 * np.arange(max_iterations)
    plt.semilogy(cost, distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt):
    cost = c1 * (np.arange(max_iterations) * t_values[index]) + c2 * np.arange(max_iterations)
    plt.semilogy(cost, distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(optimization_error_list_near_dgdt_ls):
    cost = c1 * (np.arange(max_iterations) * t_values[index]) + c2 * np.arange(max_iterations)
    plt.semilogy(cost, distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])
plt.title('Relative Error vs. Cost')
plt.xlabel('Cost (c1*Communications + c2*Gradient Evaluations)')
plt.ylabel('Log of Normalized Distance to Optimal Solution')
plt.grid(True)

# Place a single legend outside the bottom of the plots
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True,  ncol=3)

# Adjust the layout to make space for the larger plot and legend
plt.tight_layout()

# Show the plot
plt.show()


#########################
#### Consensus Error ####
#########################

colors = ['r', '#FFA500', 'c', 'b']

# Plotting all results
plt.figure(figsize=(15, 12))

# Relative Error vs. Number of Iterations
plt.subplot(2, 2, 1)
for index, distances in enumerate(consensus_error_list_dgdt):
    plt.semilogy(distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt):
    plt.semilogy(distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt_ls):
    plt.semilogy(distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])
plt.title('Consensus Error vs. Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Log of Consensus Error')
plt.grid(True)

# Relative Error vs. Number of Communications
plt.subplot(2, 2, 2)
for index, distances in enumerate(consensus_error_list_dgdt):
    plt.semilogy(np.arange(max_iterations) * t_values[index], distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt):
    plt.semilogy(np.arange(max_iterations) * t_values[index], distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt_ls):
    plt.semilogy(np.arange(max_iterations) * t_values[index], distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])
plt.title('Consensus Error vs. Number of Communications')
plt.xlabel('Number of Communications')
plt.ylabel('Log of Consensus Error')
plt.grid(True)

# Relative Error vs. Number of Gradient Evaluations
plt.subplot(2, 2, 3)
for index, distances in enumerate(consensus_error_list_dgdt):
    plt.semilogy(distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt):
    plt.semilogy(distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt_ls):
    plt.semilogy(distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])
plt.title('Consensus Error vs. Number of Gradient Evaluations')
plt.xlabel('Number of Gradient Evaluations')
plt.ylabel('Log of Consensus Error')
plt.grid(True)

# Relative Error vs. Cost
plt.subplot(2, 2, 4)
for index, distances in enumerate(consensus_error_list_dgdt):
    cost = c1 * (np.arange(max_iterations) * t_values[index]) + c2 * np.arange(max_iterations)
    plt.semilogy(cost, distances, label=f'DGDt t={t_values[index]}', linestyle='--', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt):
    cost = c1 * (np.arange(max_iterations) * t_values[index]) + c2 * np.arange(max_iterations)
    plt.semilogy(cost, distances, label=f'NEAR-DGDt t={t_values[index]}', linestyle=':', color=colors[index])
for index, distances in enumerate(consensus_error_list_near_dgdt_ls):
    cost = c1 * (np.arange(max_iterations) * t_values[index]) + c2 * np.arange(max_iterations)
    plt.semilogy(cost, distances, label=f'NEAR-DGDt-LS t={t_values[index]}', color=colors[index])
plt.title('Consensus Error vs. Cost')
plt.xlabel('Cost (c1*Communications + c2*Gradient Evaluations)')
plt.ylabel('Log of Consensus Error')
plt.grid(True)

# Place a single legend outside the bottom of the plots
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True,  ncol=3)

# Adjust the layout to make space for the larger plot and legend
plt.tight_layout()

# Show the plot
plt.show()