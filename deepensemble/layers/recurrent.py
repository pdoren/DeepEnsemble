from .layer import Layer

import theano.tensor as T
from theano import config, shared, scan
import numpy as np

__all__ = ['RecurrentLayer']


class RecurrentLayer(Layer):

    def __init__(self, n_input=None, n_recurrent=None, activation=None):
        input_shape = n_input if isinstance(n_input, tuple) else (None, n_input)
        recurrent_shape = n_recurrent if isinstance(n_recurrent, tuple) else (None, n_recurrent)
        super(RecurrentLayer, self).__init__(input_shape=input_shape,
                                             output_shape=recurrent_shape,
                                             non_linearity=activation)

        self._Wr = None

    def initialize_parameters(self):
        """ Initialize neurons params of layers
        """
        super(RecurrentLayer, self).initialize_parameters()
        if self._Wr is None:
            self._Wr = shared(np.zeros(shape=self.get_shape_W(), dtype=config.floatX), name='Wr', borrow=True)

        self._Wr.set_value(self.init_W(self.get_shape_Wr()))

        self._params = [self._W, self._Wr, self._b]

    def get_shape_Wr(self):
        """ Gets shape weights of layer.
        """
        return self.get_fan_out(), self.get_fan_out()

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        return self.get_fan_in(), self.get_fan_out()

    def get_shape_b(self):
        """ Gets shape bias of layer.
        """
        return self.get_fan_out(),

    def _step(self, input_t, previous):
        """ Step scan """
        lin_output = T.cast(T.dot(previous, self._Wr) + input_t, dtype=config.floatX)

        # noinspection PyCallingNonCallable
        return (
            lin_output if self._non_linearity is None
            else self._non_linearity(lin_output)
        )

    def output(self, x):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation:

            .. math::  Layer(x) = activation(Wx + b)
        """
        if x.ndim > 2:
            x = x.flatten(2)

        x_W_b = T.dot(x, self._W) + self._b.dimshuffle('x', 0)
        result, updates = scan(self._step,
                               sequences=[x_W_b],
                               outputs_info=[T.zeros_like(self._b)])
        return result
