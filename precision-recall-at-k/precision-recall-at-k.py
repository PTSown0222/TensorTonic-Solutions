import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # top k in a recommended list
    get_top_k = recommended[:k]
    set_top_k = set(get_top_k)
    set_relevant = set(relevant)

    hits = set_top_k.intersection(set_relevant)
    num_hits = len(hits)

    # Precision@K
    if k == 0:
        precision_at_k = 0
    else:
        precision_at_k = num_hits / k
    
    # Recall@K
    total_rev_items = len(set_relevant) 
    if total_rev_items == 0:
        recall_at_k = 0.0
    else:
        recall_at_k = num_hits / total_rev_items
    return [precision_at_k, recall_at_k]

    
    