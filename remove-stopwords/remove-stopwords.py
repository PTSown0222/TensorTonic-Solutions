def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    vector = []
    flag = 0
    for token in tokens:
        if token not in stopwords:
            vector.append(token)
            flag = 1
    if flag:      
        return vector
    return vector