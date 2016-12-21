from .layer import Layer

import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = ['Dropout']


class Dropout(Layer):

    def __init__(self, input_shape=None, p=0.5, rescale=True, seed=13):
        super(Dropout, self).__init__(input_shape=input_shape, output_shape=input_shape, exclude_params=True)

        self._p = p
        self._rescale = rescale
        self._srng = RandomStreams(seed=seed)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        self._output_shape = shape

    def output(self, x, prob=True):
        if self._p <= 0.0 or self._p > 1.0:
            return x
        else:

            one = T.constant(1.0) if isinstance(x, T.TensorVariable) else 1.0

            prob = one - self._p
            if self._rescale:
                x /= prob

            input_shape = self.get_input_shape()
            if any(s is None for s in input_shape):
                input_shape = x.shape

            return x * self._srng.binomial(input_shape, p=prob, dtype=config.floatX)
