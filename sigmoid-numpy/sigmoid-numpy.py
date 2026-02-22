import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    return np.where(x < 0,
                   np.exp(x) / (1 + np.exp(x)),
                   1 / (1 + np.exp(-x)))