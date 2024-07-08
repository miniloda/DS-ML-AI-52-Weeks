import numpy as np
import matrix 
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def coef_Mat(self, X,Y):
        X_mat = np.column_stack((np.ones(X.shape[0]), X))
        y_mat = np.array(Y)
        X_trans = matrix.transpose(X_mat)
        XtX = matrix.dot(X_trans, X_mat)
        Xty = np.dot(X_trans, Y)

        return np.dot(np.linalg.inv(XtX), Xty)
    def fit(self, X, Y):
        self.coef_mat = self.coef_Mat(X,Y)
        self._coef = self.coef_mat[1:]
        self._intercept = self.coef_mat[0]

    def predict(self,X):
        return np.dot(np.column_stack((np.ones(X.shape[0]), X)), self.coef_mat)