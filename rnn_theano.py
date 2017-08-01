import math.sqrt as sqrt
import numpy as np
import theano.tensor as T

class RNN:

    def __init__(self, vocab_size, state_size, mini_batch_size):
        self.vocab_size = vocab_size
        self.state_size = state_size
        # each matrix in x and y are input and output sequences respectively
        self.x = T.tensor3("x")
        self.y = T.tensor3("y")
        self.o = T.tensor3("o")
        # U maps an input element into an array of size state_size
        u_shape = (state_size, vocab_size)
        self.u = theano.shared((np.random.uniform(low=-1/sqrt(vocab_size),
                                                  high=1/sqrt(vocab_size)
                                                  shape=u_shape)),
                                dtype=theano.config.floatX,
                                name="u",
                                borrow=True)
        # W maps an state element into an array of the same size
        w_shape = (state_size, state_size)
        self.w = theano.shared((np.random.uniform(low=-1/sqrt(state_size),
                                                  high=1/sqrt(state_size)
                                                  shape=w_shape)),
                                dtype=theano.config.floatX,
                                name="w",
                                borrow=True)
        # V maps an element of state size into an array of size vocab_size
        v_shape = (vocab_size, state_size)
        self.v = theano.shared((np.random.uniform(low=-1/sqrt(state_size),
                                                  high=1/sqrt(state_size)
                                                  shape=v_shape)),
                                dtype=theano.config.floatX,
                                name="v",
                                borrow=True)
