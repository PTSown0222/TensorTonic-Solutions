import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # handle empty nodes
    if len(y) == 0:
        return 0.0
    eps = 1e-9
    classes, counts = np.unique(y, return_counts = True)
    H = 0.0
    # loop solution
    total_elements = len(y)
    for count in counts:
        p_i = count / total_elements
        H -= p_i * np.log2(p_i + eps)
    return np.abs(H)
        