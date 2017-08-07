import math
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import softmax

class RNN:

    def __init__(self, vocab_size, state_size):
        self.vocab_size = vocab_size
        self.state_size = state_size
        # x is an input, y is the corresponding label
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        # s is the state matrix
        # todo, 1 -> mini_batch_size?
        self.s = theano.shared(np.zeros((state_size, 1), dtype=theano.config.floatX),
                               name="s",
                               borrow=True)
        # U maps an input element into an array of size state_size
        u_shape = (state_size, vocab_size)
        uniform_u = np.array(np.random.uniform(low=-1/math.sqrt(vocab_size),
                                               high=1/math.sqrt(vocab_size),
                                               size=u_shape),
                             dtype=theano.config.floatX)
        self.u = theano.shared(uniform_u, name="u", borrow=True)
        # W maps an state element into an array of the same size
        w_shape = (state_size, state_size)
        uniform_w = np.array(np.random.uniform(low=-1/math.sqrt(state_size),
                                               high=1/math.sqrt(state_size),
                                               size=w_shape),
                             dtype=theano.config.floatX)
        self.w = theano.shared(uniform_w, name="w", borrow=True)
        # V maps an element of state size into an array of size vocab_size
        v_shape = (vocab_size, state_size)
        uniform_v = np.array(np.random.uniform(low=-1/math.sqrt(state_size),
                                               high=1/math.sqrt(state_size),
                                               size=v_shape),
                             dtype=theano.config.floatX)
        self.v = theano.shared(uniform_v, name="v", borrow=True)
        self.params = [self.U, self.W, self.V]
        # state updates are constructed from the input and the previous state
        self.update_state = (self.s, tanh(T.dot(self.u, self.x) + T.dot(self.w, self.s)))
        # the output is constructed from the state
        self.pre_o = T.dot(self.v, self.s)
        self.o = softmax(T.dot(self.v, self.s))



    # convert a list of indices into a matrix of one-hot vectors
    # i.e. process a single sentence as an input matrix
    def process_input(self, inpt):
        num_time_steps = len(inpt)
        a = np.zeros((self.vocab_size, num_time_steps))
        for i, num in enumerate(inpt):
            a[num][i] = 1
        return a

    # convert a matrix of one-hot vectors into an array of indices
    def process_output(self, output):
        output_list = []
        for i in output:
            output_list.append(np.argmax(output[i]))
        return np.array(output_list)

    def sgd(self, x_train, y_train, learning_rate, num_epochs):

        cost = ...
        grad = ...
        update_parameters = [(param, param - learning_rate * grad)
                            for param in self.params for grad in grads]

        self.train = theano.function(x, y, learning_rate, updates=[update_parameters]):

        # actually perform training
        num_examples = len(x_train)
        for epoch_i in range(num_epochs):
            # shuffle the training data
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]





    def predict(self, inpt):
        # construct theano function for prediction
        predict_ = theano.function([self.x], [self.o, self.pre_o])

        # test
        print(inpt)
        inpt_mat = self.process_input(inpt)
        print(inpt_mat)
        inpt_vector = inpt_mat[:,0]
        inpt_vector = np.reshape(inpt_vector, (15, 1))
        print(inpt_vector.shape)
        output_mat = predict_(inpt_vector)
        print(output_mat)
        # print(self.process_output(output_mat))
