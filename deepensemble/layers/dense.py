import theano.tensor as T
from .layer import Layer

__all__ = ['Dense']


class Dense(Layer):
    """ Typical Layer of MLP.

    .. math:: Layer(x) = activation(Wx + b)

    where :math:`x \\in \\mathbb{R}^{n_{output}}`, :math:`W \\in \\mathbb{R}^{n_{output} \\times n_{input}}` and
    :math:`b \\in \\mathbb{R}^{n_{output}}`.

    Parameters
    ----------
    n_input : int or tuple[]
        Dimension of input.

    n_output : int or tuple[]
        Dimension of output.

    activation : callback
        Activation function.
    """

    def __init__(self, n_input=None, n_output=None, activation=None):
        input_shape = n_input if isinstance(n_input, tuple) else (None, n_input)
        output_shape = n_output if isinstance(n_output, tuple) else (None, n_output)
        super(Dense, self).__init__(input_shape=input_shape, output_shape=output_shape, non_linearity=activation)

    def output(self, x, prob=True):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        prob : bool
            Flag for changing behavior of some layers.

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation:

            .. math::  Layer(x) = activation(Wx + b)
        """
        if x.ndim > 2:
            x = x.flatten(2)

        lin_output = T.dot(x, self.get_W()) + self.get_b().dimshuffle('x', 0)

        # noinspection PyCallingNonCallable
        return (
            lin_output if self._non_linearity is None
            else self._non_linearity(lin_output)
        )
