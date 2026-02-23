import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = Q @ W_q  # Shape: (batch, seq_len, d_model)
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_split = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_split = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_split = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    K_transposed = K_split.transpose(0, 1, 3, 2) # Shape: (batch_size, num_heads, d_k, seq_length)
    attn_score = (Q_split @ K_transposed)/np.sqrt(d_k)
    attn_prob = softmax(attn_score)
    context = attn_prob @ V_split
    context_concat = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    output = context_concat @ W_o 
    return output
    
    

    
    

    
    