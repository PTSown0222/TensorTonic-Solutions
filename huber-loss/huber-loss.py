import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    error = y_true - y_pred
    huberLoss = np.where(np.abs(error)> delta, delta * (np.abs(error) - 1/2 * delta), 1/2 * error **2)
    huberLossAvg = np.mean(huberLoss)
    return huberLossAvg