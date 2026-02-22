import numpy as np
def _tanh(x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    z = np.dot(x_t,Wx) + np.dot(h_prev, Wh) + b
    h_t = _tanh(z)
    return h_t
