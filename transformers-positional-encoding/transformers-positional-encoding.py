import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_indices = np.arange(seq_length).reshape(-1,1) # (seq_length,1)
    pe = np.zeros((seq_length, d_model))
    
    div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)

    pe[:,0::2] = np.sin(pos_indices / div_term)
    pe[:,1::2] = np.cos(pos_indices / div_term)

    return pe

    
    