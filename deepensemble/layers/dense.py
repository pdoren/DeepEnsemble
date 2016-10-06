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
    n_input : int
        Dimension of input.

    n_output : int
        Dimension of output.

    activation : callback
        Activation function.
    """

    def __init__(self, n_input=None, n_output=None, activation=None):
        super(Dense, self).__init__(input_shape=(n_input,), output_shape=(n_output,), non_linearity=activation)

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        return self.get_fan_in(), self.get_fan_out()

    def get_shape_b(self):
        """ Gets shape bias of layer.
        """
        return self.get_fan_out(),

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
        lin_output = T.dot(x, self._W) + self._b

        # noinspection PyCallingNonCallable
        return (
            lin_output if self._non_linearity is None
            else self._non_linearity(lin_output)
        )
