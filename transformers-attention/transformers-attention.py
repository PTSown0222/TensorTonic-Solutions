import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = K.shape[-1]
    score = Q @ K.transpose(-2,-1)
    attn_score = score/math.sqrt(d_k)
    attn_prob = F.softmax(attn_score, dim=-1)
    output = attn_prob @ V
    return output
    
    
    