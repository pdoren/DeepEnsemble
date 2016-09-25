import theano.tensor as T
from theano import shared, config
import numpy as np
from .layer import Layer

__all__ = ['Dense']


class Dense(Layer):
    """ Typical Layer of MLP.

    .. math:: Layer(x) = activation(Wx + b)

    where :math:`x \\in \\mathbb{R}^{n_{output}}`, :math:`W \\in \\mathbb{R}^{n_{output} \\times n_{input}}` and
    :math:`b \\in \\mathbb{R}^{n_{output}}`.

    Attributes
    ----------
    __activation : theano.Op
        Activation functions.

    __W : theano.shared
        Weight of neurons.

    __b : theano.shared
        Bias.

    _params : list
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
        super(Dense, self).__init__(n_input=n_input, n_output=n_output)
        self.__activation = activation
        self.__W = None
        self.__b = None

    def get_W(self):
        """ Get parameters of Layer.

        Returns
        -------
        theano.tensor.matrix
            Returns parameter W of layer.
        """
        return self.__W

    def initialize_parameters(self):
        """ Initialize neurons params of layers
        """
        if self.__W is None:
            self.__W = shared(np.zeros(shape=(self._n_input, self._n_output), dtype=config.floatX), name='W',
                              borrow=True)
        if self.__b is None:
            self.__b = shared(np.zeros(shape=(self._n_output,), dtype=config.floatX), name='b',
                              borrow=True)
        self._params = [self.__W, self.__b]

        W = np.array(np.random.uniform(low=-np.sqrt(6.0 / (self._n_input + self._n_output)),
                                       high=np.sqrt(6.0 / (self._n_input + self._n_output)),
                                       size=(self._n_input, self._n_output)), dtype=config.floatX)
        if self.__activation == T.nnet.sigmoid:
            W *= 4

        self.__W.set_value(W)
        self.__b.set_value(np.zeros(shape=(self._n_output,), dtype=config.floatX))

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
        lin_output = T.dot(x, self.__W) + self.__b

        # noinspection PyCallingNonCallable
        return (
            lin_output if self.__activation is None
            else self.__activation(lin_output)
        )
