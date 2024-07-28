import numpy as np


class LinearRegression:

    def __init__(self, max_iter=1000):
        self.intercept_ = None
        self.coef_ = None
        self.J_arr = None
        self.max_iter = max_iter

    def fit(self, X, y):
        self.coef_, self.intercept_, self.J_arr = gradient_descent(X, y, self.max_iter)

    def predict(self, X):
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        return np.dot(X, self.coef_.T) + self.intercept_


def abs_distance(w_old, w_new):
    return np.linalg.norm(w_new - w_old)


def gradient_descent(X, y, max_iter, b=0, a=0.1, err=0.0001):
    """
    Performs gradient descent to find the optimal weights and bias for linear regression.

    Parameters:
    X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    y (numpy.ndarray): Target values of shape (m,).
    max_iter (int): Maximum number of iterations for gradient descent.
    b (float): Initial bias. Default is 0.
    a (float): Learning rate. Default is 0.1.
    err (float): Error tolerance for convergence. Default is 0.0001.

    Returns:
    tuple: A tuple containing the optimal weights (w_arr), the optimal bias (b), and the cost values at each iteration (J_arr).
    """
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
    assert len(y.shape) == 1, "y must be a 1D array"
    assert len(X.shape) == 2, "X must be a 2D array. If you have a single feature, reshape it to (-1, 1)" 
    m, n = X.shape
    # Initialize weights
    w_arr_old = np.random.randn(n) * 0.1
    w_arr = np.zeros(X.shape[1], dtype=np.float64)
    J_arr = np.zeros(max_iter + 1)
    y = np.array(y, dtype=float)
    b_old = b
    for i in range(max_iter):
        J_arr[i] = compute_cost(X, y, w_arr_old, b)
        dj_dw, dj_db = compute_gradient(X, y, w_arr_old, b)

        w_arr[:] = w_arr_old[:] - np.dot(a, dj_dw)
        b = b_old - a * dj_db
        # assign the current weights to old weights for the next iteration
        w_arr_old = w_arr.copy()
        b_old = b
        # Check if the weights have converged
        if abs_distance(J_arr[i], J_arr[i - 1]) <= err:
            print("Converged at iteration", i)
            return w_arr, b, J_arr

    print("Did not converge after", max_iter, "iterations", ". Try increasing max_iter.")
    return w_arr, b, J_arr


def compute_cost(X, y, w, b ):
    """
        Computes the cost function for linear regression.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
        y (numpy.ndarray): Target values of shape (m,).
        w (numpy.ndarray): Current weights of shape (n,).
        b (float): Current bias.

        Returns:
        float: The cost value at the current iteration.
        """
    m = X.shape[0]
    return np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)


def compute_gradient(X, y, w_arr, b):
    """
        Computes the gradient of the cost function for linear regression.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
        y (numpy.ndarray): Target values of shape (m,).
        w_arr (numpy.ndarray): Current weights of shape (n,).
        b (float): Current bias.
        J_arr (numpy.ndarray): Array to store cost values at each iteration.

        Returns:
        tuple: A tuple containing the gradient of weights (dj_dw) and the gradient of bias (dj_db).
        """
    m = X.shape[0]
    dj_dw = np.dot(np.dot(X, w_arr) + b * np.ones(m) - y, X) / m
    dj_db = np.sum(np.dot(X, w_arr) + b * np.ones(m) - y) / m
    return dj_dw, dj_db
