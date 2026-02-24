import math
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    dot_product = sum( i * j for i, j in zip(x1,x2))
    norms1 = math.sqrt(sum(i * i for i in x1))
    norms2 = math.sqrt(sum(j * j for j in x2))

    # avoid div by 0, I use eps
    eps = 1e-8 
    cosine = dot_product / (norms1 * norms2 + eps)

    if label == 1:
        loss = float(1 - cosine)
    elif label == -1:
        loss = float(max(0, cosine - margin))
    else:
        return
    return loss