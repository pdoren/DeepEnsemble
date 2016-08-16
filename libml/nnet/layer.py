import theano.tensor as T
from theano import shared, config
import numpy as np


class Layer:
    def __init__(self, n_input, n_neurons, activation=None):
        """ Typical Layer of MLP

        Parameters
        ----------
        n_input
        n_neurons
        activation
        """
        self.N_input = n_input
        self.N_neurons = n_neurons
        self.activation = activation
        self.W = shared(np.zeros(shape=(n_input, n_neurons), dtype=config.floatX), name='W', borrow=True)
        self.b = shared(np.zeros(shape=(n_neurons,), dtype=config.floatX), name='b', borrow=True)
        self.initialize_parameters()
        self.params = [self.W, self.b]

    def initialize_parameters(self):
        """ Initialize neurons params of layer

        Returns
        -------

        """
        W = np.array(np.random.uniform(low=-np.sqrt(6.0 / (self.N_input + self.N_neurons)),
                                       high=np.sqrt(6.0 / (self.N_input + self.N_neurons)),
                                       size=(self.N_input, self.N_neurons)), dtype=config.floatX)
        if self.activation == T.nnet.sigmoid:
            W *= 4

        self.W.set_value(W)
        self.b.set_value(np.zeros(shape=(self.N_neurons,), dtype=config.floatX))

    def output(self, x):
        """ Return output of layer

        Parameters
        ----------
        x

        Returns
        -------

        """
        lin_output = T.dot(x, self.W) + self.b

        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
