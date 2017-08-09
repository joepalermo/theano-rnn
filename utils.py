import pickle
import numpy as np

def pp_output(a, index_to_word):
    """Convert an array of indices into a string containing the corresponding
    words."""
    return " ".join([index_to_word[i] for i in a])

def save_model_parameters(model, save_params_as):
    model_params = dict()
    model_params['U'] = model.U.get_value()
    model_params['W'] = model.W.get_value()
    model_params['V'] = model.V.get_value()
    pickle.dump(model_params, open(save_params_as, 'wb'))

def load_model_parameters(path_to_params):
    return pickle.load(open(path_to_params, 'rb'))
