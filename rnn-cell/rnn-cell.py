import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    # x_t.T @ Wxh (4,2)
    # h_prev.T @ Whh (4,2)
    # b_h (4,)
    z = np.dot(x_t, W_xh.T) + np.dot(h_prev, W_hh.T) + b_h
    h_t = np.tanh(z)
    return h_t