def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    vocab_dict = {}
    for phrase in sentences:
        for word in phrase:
            vocab_dict[word] = vocab_dict.get(word,0) + 1
    return vocab_dict