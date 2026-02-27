import numpy as np
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X = np.asanyarray(X)
    W = np.asanyarray(W)
    b = np.asanyarray(b)
    
    num_samples = X.shape[0]
    num_features_out = W.shape[1]
    Y = np.zeros((num_samples, num_features_out))
    
    for i in range(num_samples):
        for j in range(num_features_out):
            Y[i, j] = np.sum(X[i, :] * W[:, j]) + b[j]
    return Y.tolist()