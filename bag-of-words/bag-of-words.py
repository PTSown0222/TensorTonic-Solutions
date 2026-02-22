import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    # init a vector zeros with len of vocab
    vector = np.zeros(len(vocab), dtype=int)
    # create a dict to retrive with O(1)
    vocab_dict = {word : index for index, word in enumerate(vocab)}

    for token in tokens:
        ind = vocab_dict.get(token) # get ind of token in vocab_dict
        if ind is not None:
            vector[ind] += 1

    return vector
        
    