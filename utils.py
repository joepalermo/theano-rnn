import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def pp_output(a, index_to_word):
    """Convert an array of indices into a string containing the corresponding
    words."""
    return " ".join([index_to_word[i] for i in a])
