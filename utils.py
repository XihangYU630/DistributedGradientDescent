import numpy as np

def generate_matrix_with_condition_number(p, condition_number):
    """
    Generate a positive definite matrix of size p x p with a specific condition number.
    """
    # Generate a random matrix
    M = np.random.randn(p, p)
    U, s, V = np.linalg.svd(M)  # Singular value decomposition
    
    # Adjust the singular values
    max_singular = 1.0  # Scale of the largest singular value
    min_singular = max_singular / condition_number
    s_new = np.linspace(min_singular, max_singular, num=p)
    
    # Construct the diagonal matrix of singular values
    S_new = np.diag(s_new)
    
    # Reconstruct the matrix with the new singular values
    A = U @ S_new @ V.T
    A = A @ A.T  # Ensure the matrix is positive definite
    print("Condition Number:", np.linalg.cond(A))

    return A

def generate_list_of_matrices(n, p, condition_number):
    """
    Generate a list of n positive definite matrices, each of size p x p,
    with a specified condition number.

    Parameters:
    - n: Number of matrices (int).
    - p: Dimension size of each matrix (int).
    - condition_number: Desired condition number for each matrix (float).

    Returns:
    - A_list: List of matrices (list of numpy arrays).
    """
    return [generate_matrix_with_condition_number(p, condition_number) for _ in range(n)]


def distributed_quadratic_gradient(x, A, b):
    """
    Compute the distributed gradient of the quadratic function.

    Parameters:
    - x: Current point (numpy array of shape (n*p,)).
    - A: List of matrices A_i for each node (list of numpy matrices).
    - b: List of vectors b_i for each node (list of numpy arrays).

    Returns:
    - gradient: The computed gradient (numpy array of shape (n*p,)).
    """
    n = len(A)
    p = A[0].shape[0]
    gradient = np.zeros(n * p)
    for i in range(n):
        gradient[i*p:(i+1)*p] = A[i] @ x[i*p:(i+1)*p] + b[i]
    return gradient



def create_4_cyclic_graph(n):
    """
    Create the weight matrix for a 4-cyclic graph where each node is connected to its four immediate neighbors.

    Parameters:
    - n: Total number of nodes (int).

    Returns:
    - W: Weight matrix (numpy array of shape (n, n)).
    """
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i-2) % n] = 1  # Connect to 2nd previous node
        W[i, (i-1) % n] = 1  # Connect to previous node
        W[i, (i+1) % n] = 1  # Connect to next node
        W[i, (i+2) % n] = 1  # Connect to 2nd next node
    W /= np.sum(W, axis=1, keepdims=True)  # Normalize to make the rows sum to 1
    return W

def compute_global_optimal_solution(A_list, b_list):
    """
    Compute the global optimal solution for the distributed quadratic problem.

    - A_list: List of matrices A_i for each node.
    - b_list: List of vectors b_i for each node.

    Returns:
    - x_opt: The optimal solution vector.
    """
    # Sum up all A_i matrices and b_i vectors
    A_sum = sum(A_list)
    b_sum = sum(b_list)
    # Solve for x in the equation sum(A_i)x = -sum(b_i)
    x_opt = np.linalg.solve(A_sum, -b_sum)
    return x_opt

def evaluate_quadratic_function(x, A, b):
    """
    Evaluates the sum of quadratic functions across nodes in a distributed system.
    
    Args:
    - x (numpy.ndarray): The vector x for which the function is evaluated, shape (np, 1).
    - A (list of numpy.ndarray): List of symmetric matrices A_i for each node, each A_i is (p, p).
    - b (list of numpy.ndarray): List of vectors b_i for each node, each b_i is (p, 1).
    
    Returns:
    - float: The value of the quadratic function f(x) = sum(1/2 * x_i.T * A_i * x_i - b_i.T * x_i).
    """
    n = len(A)  # number of nodes
    p = b[0].shape[0]  # size of each node's vector
    total_function_value = 0
    
    for i in range(n):
        x_i = x[i*p:(i+1)*p]  # slice out the segment of x for node i
        A_i = A[i]
        b_i = b[i]
        
        term1 = 0.5 * np.dot(x_i.T, np.dot(A_i, x_i))  # 1/2 * x_i^T * A_i * x_i
        term2 = np.dot(b_i.T, x_i)  # b_i^T * x_i
        total_function_value += (term1 + term2)
        
    return total_function_value