import pickle
import numpy as np
from datetime import datetime

def pp_output(output, index_to_word):
    """Convert an array of indices into a string containing the corresponding
    words."""
    return " ".join([index_to_word[i] for i in output])

def save_model_parameters(model, model_dir):
    save_params_as = model_dir + "/best_model"
    model_params = dict()
    model_params['U'] = model.U.get_value()
    model_params['W'] = model.W.get_value()
    model_params['V'] = model.V.get_value()
    pickle.dump(model_params, open(save_params_as, 'wb'))

def load_model_parameters(model_path):
    return pickle.load(open(model_path, 'rb'))
