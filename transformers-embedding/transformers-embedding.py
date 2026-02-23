import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # embd_matrix (vocab_size, d_embd)
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    n_embds = embedding(tokens)
    scaled_embeds = n_embds * math.sqrt(d_model)
    return scaled_embeds
    
    