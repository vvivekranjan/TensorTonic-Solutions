import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    res = list()
    for val in vocab:
        res.append(tokens.count(val))
    return np.array(res, dtype='int')
    pass