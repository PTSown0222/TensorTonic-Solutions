import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # X (N,M) -> rows=N, cols = M
    # w (M), b = 0
    rows, cols = X.shape
    w = np.zeros(cols)
    b = 0.0
    for _ in range(steps):
        z = np.dot(X,w) + b
        p = _sigmoid(z)
        
        dloss = p - y
        dw = (1/rows) * np.dot(X.T, dloss)
        db = (1 / rows) * np.sum(dloss)
        w -= dw * lr 
        b -= db * lr
        
    return w, b