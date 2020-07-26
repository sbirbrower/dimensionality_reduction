import numpy as np
import pandas as pd


def neg_eucl_dist(X):
    """
    Find negative squared euclidean distance between all points/rows in X
    for points 1 and 2 and dimensions x, y, z, etc:
        distance = (x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2 + ... 
    and do this for all points.
    returns result: result_ij is the negative distance between rows X_i and X_j
    """
    result = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        result[i] = np.sum((X - X[i])**2, axis=1)
    return -1 * result    


def binary_search(f, target, tolerance=1e-8, max_iter=10000, lower=1e-15, upper=1000):
    """
    given f(x) = target, returns x
    """

    for i in range(max_iter):
        mid = (lower + upper) / 2
        y = f(mid)
        if y > target:
            upper = mid
        else:
            lower = mid
        if tolerance >= np.abs(y - target):
            break
    return mid

def cal_perplexity(X, row_no):
    """
    X : a matrix row that represents negative euclidean distance divided by the constant 2 * np.square(sigma)
    row_no : used to set diagonal p_ij = p_ji = 0
    returns : 2^{- \sum_j p_{j|i} log_2 p_{j|i}}
    """
    e_x = np.exp(X - np.max(X))  # first calculate the conditional probabilities p
    e_x[row_no] = 0   # only interested in modeling pairwise similarities, set diagnal of P matrix to zero
    e_x = np.maximum(e_x, 1e-8)  # avoid taking the log of zero
    entropy = -np.sum((e_x / np.sum(e_x)) * np.log2(e_x / np.sum(e_x)))
    return 2**entropy

def calc_sigmas(X, desired_perplexity):
    """
    X : negative euclidean distance matrix
    desired_perplexity : input by user, typically between 5 and 50
    returns array of correct sigmas for each row, where sigma is the variance of the Gaussian that is 
    centered on datapoint x_i
    a smaller sigma indicates that a point is located in a dense region
    """
    sigmas = []
    for i in range(X.shape[0]):
        perplexity_fn = lambda sigma: cal_perplexity(X[i]/(2 * np.square(sigma)), row_no=i)
        
        correct_sigma = binary_search(perplexity_fn, desired_perplexity)
        
        sigmas.append(correct_sigma)
        
    return np.array(sigmas)
    
def calc_p_ji(data, desired_perplexity):
    """
    data : original data, rows are datapoints, columns are dimensions
    returns P : the joint probability matrix
    P_{j|i} = \frac{exp (- ||x_i - x_j||^2 / 2 \sigma_i^2)} {\sum_k exp (- ||x_i - x_k||^2 / 2 \sigma_i^2)}
    """
    X = neg_eucl_dist(data)
    sigmas = calc_sigmas(X, desired_perplexity)
    
    X = X / (2 * np.square(np.array(sigmas).reshape((-1, 1)))) 
    e_x = np.exp(X - np.max(X, axis=1))
    np.fill_diagonal(e_x, 0)
    e_x = np.maximum(e_x, 1e-8)
    P = e_x / np.sum(e_x, axis=1)
    
    # make P_{j|i} matrix symmetric, take average of distances
    return (P + P.T) / (2 * P.shape[0])

def calc_q_ji(Y):
    """
    Y : matrix in two dimensional space with n rows
    returns : matrix of conditional probabilites, modeling the similaries between two points in Y
    """
    X = neg_eucl_dist(Y)
    
    inv_x = np.power(1 - X, -1)
    np.fill_diagonal(inv_x, 0)
    Q = inv_x / np.sum(inv_x)
    
    return Q

def calc_grad(P, Q, Y):
    """
    calculates gradient
    4 * \sum_j (p_{ij} - p_{ij}) (y_i - y_j) (1 + ||y_i - y_j||)^-1
    """
    n = P.shape[0]
    Y_sums = np.zeros((n, 2))
    Y_diff = np.zeros((n, n, 2))

    pq_sums = np.zeros((n, n))
    pq_diff = P - Q
    
    inv_distances = np.power(1 - neg_eucl_dist(Y), -1)

    for i in range(n):
        pq_sums[i] = pq_diff[i]
        for j in range(n):
            Y_sums[j] = Y[i] - Y[j] 
            Y_diff[i] = Y_sums
    dCdy = 4 * np.sum( pq_sums[..., np.newaxis] * Y_diff * inv_distances[..., np.newaxis], axis=1)
    return dCdy

def t_sne(X, desired_perplexity=20, num_iters=500, learning_rate=10, print_update=True):
    """
    computes t-sne
    input : data matrix X
    desired_perplexity : input by user, typically between 5 and 50
    """
    P = calc_p_ji(X, desired_perplexity)
    Y = np.random.randn(X.shape[0], 2)

    for i in range(num_iters):  # gradient descent
        Q = calc_q_ji(Y)
        grads = calc_grad(P, Q, Y)
        Y = Y - learning_rate * grads
        
        if print_update:
            if (i + 1) % 25 == 0:
                error = np.sum(P * np.log(P / np.maximum(Q, 1e-12)))
                print(f"Iteration {i+1}: error is {error}")
    return Y