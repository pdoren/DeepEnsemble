import theano.tensor as T
from theano import shared, config
import numpy as np


class Dense:
    """ Typical Layer of MLP.

    .. math:: Layer(x) = activation(Wx + b)

    where :math:`x \\in \\mathbb{R}^{n_{output}}`, :math:`W \\in \\mathbb{R}^{n_{output} \\times n_{input}}` and
    :math:`b \\in \\mathbb{R}^{n_{output}}`.

    Attributes
    ----------
    n_input : int
        Dimension of input.

    n_output : int
        Dimension of output.

    activation : theano.Op
        Activation functions.

    W : theano.shared
        Weight of neurons.

    b : theano.shared
        Bias.

    params : list
        Parameters of model: [W, b].

    Parameters
    ----------
    n_input : int
        Dimension of input.

    n_output : int
        Dimension of output.

    activation : theano.tensor
        Activation function.
    """

    def __init__(self, n_input=None, n_output=None, activation=None):
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.W = None
        self.b = None
        self.params = []

    def initialize_parameters(self):
        """ Initialize neurons params of layers
        """
        if self.W is None:
            self.W = shared(np.zeros(shape=(self.n_input, self.n_output), dtype=config.floatX), name='W', borrow=True)
        if self.b is None:
            self.b = shared(np.zeros(shape=(self.n_output,), dtype=config.floatX), name='b', borrow=True)
        self.params = [self.W, self.b]

        W = np.array(np.random.uniform(low=-np.sqrt(6.0 / (self.n_input + self.n_output)),
                                       high=np.sqrt(6.0 / (self.n_input + self.n_output)),
                                       size=(self.n_input, self.n_output)), dtype=config.floatX)
        if self.activation == T.nnet.sigmoid:
            W *= 4

        self.W.set_value(W)
        self.b.set_value(np.zeros(shape=(self.n_output,), dtype=config.floatX))

    def get_parameters(self):
        """ Get parameters of Layer.

        Returns
        -------
        theano.tensor.matrix
            Returns parameter W of layer.
        """
        return self.W

    def output(self, x):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        Returns
        -------
        theano.tensor.floatX
            Returns the output layers according to above equation:

            .. math::  Layer(x) = activation(Wx + b)
        """
        lin_output = T.dot(x, self.W) + self.b

        # noinspection PyCallingNonCallable
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
