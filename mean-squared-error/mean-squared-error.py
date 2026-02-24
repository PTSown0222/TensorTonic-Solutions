import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n = len(y_pred)
    if len(y_pred) != len(y_true):
        return None
    mse = np.sum((y_pred - y_true)**2)
    mse_avg = (1 / n) * mse
    return float(mse_avg)
