import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_one_hot(value, value_range):
    """Compute a one hot vector given a value and the range of values."""
    a = np.zeros(value_range)
    a[value] = 1
    return a

def process_input(inpt_list, value_range):
    """Convert a list of indices into a matrix of one-hot vectors."""
    a = np.zeros((value_range, len(inpt_list)))
    for i, num in enumerate(inpt_list):
        a[num][i] = 1
    return a

def process_output(mat):
    """Convert a matrix of one-hot vectors into a list of indices."""
    return list(np.argmax(mat, axis=1))
