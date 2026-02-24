def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions = np.asarray(prob_distributions)
    actual_tokens = np.asarray(actual_tokens)
    
    n = len(actual_tokens)
    p_i = prob_distributions[np.arange(n), actual_tokens]
    H = - 1/n * np.sum(np.log(p_i + 1e-9))
    PP = np.exp(H)
    return float(PP)