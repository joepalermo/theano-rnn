"""This code is inspired by an RNN implementation by Danny Britz:
https://github.com/dennybritz/rnn-tutorial-rnnlm/
"""

import numpy as np
import theano as theano
import theano.tensor as T

class RNN:

    def __init__(self, vocab_size, state_size, bptt_truncate,
                 model_params=None):
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.bptt_truncate = bptt_truncate

        # define inputs to the computation graph
        x = T.ivector('x')
        y = T.ivector('y')
        eta = T.scalar('eta') # eta is the learning rate

        # if there are pre-existing model parameters, use them
        if model_params:
            U, W, V = model_params['U'], model_params['W'], model_params['V']
        else:
            print("initializing model weights")
            U = np.random.uniform(-np.sqrt(1/vocab_size), np.sqrt(1/vocab_size), (state_size, vocab_size))
            V = np.random.uniform(-np.sqrt(1/state_size), np.sqrt(1/state_size), (vocab_size, state_size))
            W = np.random.uniform(-np.sqrt(1/state_size), np.sqrt(1/state_size), (state_size, state_size))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.params = [self.U, self.V, self.W]

        # build recurrence into the computation graph
        def layer_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [outputs, states], updates = theano.scan(
            fn=layer_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.state_size))],
            non_sequences=[self.U, self.V, self.W],
            truncate_gradient=self.bptt_truncate)

        # define outputs of the computation graph
        prediction = T.argmax(outputs, axis=1)
        cost = T.sum(T.nnet.categorical_crossentropy(outputs, y))
        grads = T.grad(cost, self.params)
        gradient_descent_updates = [(param, param - eta * grad) \
                                    for param, grad in zip(self.params, grads)]

        # define functions for computing results and training
        self.propagate = theano.function([x], outputs)
        self.predict = theano.function([x], prediction)
        self.get_cost = theano.function([x, y], cost)
        self.train_on_example = theano.function(inputs=[x,y,eta],
                                                updates=gradient_descent_updates)

    def compute_cost(self, xs, ys):
        """Compute the cost with respect to a collection of data"""
        cumulative_cost = np.sum([self.get_cost(x,y) for x,y in zip(xs, ys)])
        num_examples = np.sum([len(y) for y in ys])
        return cumulative_cost/num_examples

    def sgd(self, training_data, num_epochs, learning_rate,
            validation_data=None, test_data=None):
        """Perform stochastic gradient descent."""
        x_train, y_train = training_data
        if validation_data:
            x_validation, y_validation = validation_data
        num_examples = len(x_train)
        for epoch_i in range(num_epochs):
            # shuffle the training data
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            x_train = x_train[perm]
            y_train = y_train[perm]
            print("training epoch {}".format(epoch_i))
            # update parameters once for each training example
            for i in range(num_examples):
                self.train_on_example(x_train[i],y_train[i],learning_rate)
            # compute post-epoch validation cost
            if validation_data:
                validation_cost = self.compute_cost(x_validation, y_validation)
                print("validation cost at epoch {} is: {}".format(epoch_i, validation_cost))

    def get_predictions(self, inputs):
        """Produce outputs for a set of inputs. Parameter inputs are represented
        as a list of indices."""
        return [self.predict(x) for x in inputs]
