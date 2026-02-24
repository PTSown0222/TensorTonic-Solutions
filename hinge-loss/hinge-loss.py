import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score) 
    
    hinge_l = np.maximum(0, margin - y_score * y_true)
    if reduction == "mean":
        hingeLoss = np.mean(hinge_l)
    elif reduction == "sum":
        hingeLoss = np.sum(hinge_l)
    return float(hingeLoss)
        