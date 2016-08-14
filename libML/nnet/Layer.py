import theano.tensor as T
from theano import shared, config
import numpy as np


class Layer:
    def __init__(self, n_input, n_neurons, activation=None):
        """
        Typical Layer of MLP

        :type n_input: int
        :param n_input: Dimensionality of input
        :type n_neurons: int
        :param n_neurons: Numbers of Neurons
        :type activation: tensor.Op or function
        :param activation: Non linearity to be applied in the layer
        """
        self.N_input = n_input
        self.N_neurons = n_neurons
        self.activation = activation
        self.W = shared(np.zeros(shape=(n_input, n_neurons), dtype=config.floatX), name='W', borrow=True)
        self.b = shared(np.zeros(shape=(n_neurons,), dtype=config.floatX), name='b', borrow=True)
        self.initialize_parameters()
        self.params = [self.W, self.b]

    def initialize_parameters(self):
        """
        Initialize neurons params of layer
        Note : optimal initialization of weights is dependent on the
               activation function used (among other things).
               For example, results presented in [Xavier10] suggest that you
               should use 4 times larger initial weights for sigmoid
               compared to tanh  We have no info for other function,
               so we use the same as tanh.
        """
        W = np.array(np.random.uniform(low=-np.sqrt(6.0 / (self.N_input + self.N_neurons)),
                                       high=np.sqrt(6.0 / (self.N_input + self.N_neurons)),
                                       size=(self.N_input, self.N_neurons)), dtype=config.floatX)
        if self.activation == T.nnet.sigmoid:
            W *= 4

        self.W.set_value(W)
        self.b.set_value(np.zeros(shape=(self.N_neurons,), dtype=config.floatX))

    def output(self, x):
        """
        Return output of layer

        :type x: theano.tensor.dmatrix
        :param x: a symbolic tensor of shape (n_examples, n_inputs)

        :return: return output Layer
        """
        lin_output = T.dot(x, self.W) + self.b

        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
