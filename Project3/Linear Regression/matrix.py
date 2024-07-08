import numpy as np

#TODO: ADD ASSERTIONS
def transpose(X):
    """
    Transposes a matrix
    Args:
        X ([array]): a nxm array
    """
    n = X.shape[0]
    m = X.shape[1]
    X_transpose = np.zeros(shape=(m, n))
    for i in range(m):
        for j in range(n):
            X_transpose[i][j] = X[j][i]
    return X_transpose


def dot(X, Y):
    print(X.shape)
    print(len(X.shape))
    print(Y.shape)
    print(len(Y.shape))
    if (len(X.shape) == 1 and len(Y.shape) == 1):
        return float([X[i]*Y[i] for i in len(X)])
    m = X.shape[0]
    n = Y.shape[1]
    dot_prod = np.zeros(shape = (m,n))
    for i in range(m):
        for j in range(n):
            dot_prod[i,j] = sum(X[i, k] * Y[k, j] for k in range(X.shape[1]))
    print(dot_prod)
    return dot_prod