import numpy as np


class LinearRegression:

    def __init__(self, max_iter=100000):
        self.w_arr = None
        self.b = None
        self.J_arr = None
        self.max_iter = max_iter

    def fit(self, X, y):
        self.w_arr, self.b, self.J_arr = gradient_descent(X, y, self.max_iter)

    def predict(self, X):
        return np.dot(X, self.w_arr) + self.b


def abs_distance(w_old, w_new):
    return np.linalg.norm(w_new - w_old)

def scale(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X
def gradient_descent(X, y, max_iter, b=0, a=0.1, err=0.001):
    
    m,n = X.shape
    # Initialize weights
    w_arr_old = np.random.randn(n) * 0.1
    w_arr = np.zeros(X.shape[1], dtype=np.float64)
    J_arr = np.zeros(max_iter + 1)
    X = scale(X)
    y = np.array(y, dtype=float)
    b_old = b
    for i in range(max_iter):
        J_arr[i] = compute_cost(X, y, w_arr, b, i)
        dj_dw, dj_db = compute_gradient(X, y, w_arr, b, J_arr, a, i)

        w_arr[:] = w_arr_old[:] - a * dj_dw
        b = b_old - a * dj_db
        # assign the current weights to old weights for the next iteration
        w_arr_old = w_arr.copy()
        b_old = b
        # Check if the weights have converged
        #if abs_distance(w_arr, w_arr) <= err and abs(b - b_old) <= err:
         #   print("Converged at iteration", i)
          #  return w_arr, b

    print("Did not converge after", max_iter, "iterations", ". Try increasing max_iter.")
    return w_arr, b, J_arr


def compute_cost(X, y, w, b, i):
    """
       Computes the cost function for linear regression.

       Parameters:
       X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
       y (numpy.ndarray): Target values of shape (m,).
       w (numpy.ndarray): Current weights of shape (n,).
       b (float): Current bias.
       i (int): Current iteration number.

       Returns:
       float: The cost value at the current iteration.
       """
    m = X.shape[0]
    return np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)


def compute_gradient(X, y, w_arr, b, J_arr, alpha, i):
    """
       Computes the gradient of the cost function for linear regression.

       Parameters:
       X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
       y (numpy.ndarray): Target values of shape (m,).
       w_arr (numpy.ndarray): Current weights of shape (n,).
       b (float): Current bias.
       J_arr (numpy.ndarray): Array to store cost values at each iteration.
       alpha (float): Learning rate.
       i (int): Current iteration number.

       Returns:
       tuple: A tuple containing the gradient of weights (dj_dw) and the gradient of bias (dj_db).
       """
    m = X.shape[0]
    dj_dw = np.array([np.sum([(w_arr[j]*X[i,j] - y[i])*X[i,j] for i in range(m)]) / m for j in range(X.shape[1])], dtype = np.float64)
    dj_db = np.sum([(b*X[i,j] - y[i])*X[i,j] for i in range(m) for j in range(X.shape[1])]) / m
    return dj_dw, dj_db
