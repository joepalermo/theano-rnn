def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_one_hot(value, value_range):
    """Compute a one hot vector given the value and the range of values."""
    a = np.zeros(value_range)
    a[value] = 1
    return a
