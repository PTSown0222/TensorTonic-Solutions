import numpy as np
import math
import torch

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos_indices = np.arange(seq_len).reshape(-1,1) # (seq_length,1)
    pe = np.zeros((seq_len, d_model))

    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)

    pe[:,0::2] = np.sin(pos_indices / div_term)
    
    odd_cols = d_model // 2
    pe[:,1::2] = np.cos(pos_indices / div_term)[:, :odd_cols]

    return pe

    # $e^{- \frac{2i}{d_{model}} \ln(10000)}$ --> transpose formulate
    # 