from .layer import Layer
from theano.tensor.signal.pool import pool_2d
import theano.tensor as T

__all__ = ['MaxPool1D', 'MaxPool2D', 'Pool1D', 'Pool2D']


# noinspection PyUnusedLocal
class PoolBase(Layer):
    """ Pool Base Layer

    Attributes
    ----------
    _pool_size : int or tuple[]

    _stride : int or tuple[]

    _pad : int or tuple[]

    _ignore_border : ignore_border

    _mode : mode


    Parameters
    ----------
    pool_size : int or tuple[]

    stride : int or tuple[]

    pad : int or tuple[]

    ignore_border : bool

    mode : str

    """
    def __init__(self, pool_size, stride=1, pad=0, ignore_border=True, mode='max'):
        super(PoolBase, self).__init__(input_shape=None, output_shape=None, non_linearity=None, exclude_params=True)

        self._pool_size = pool_size
        self._stride = stride
        self._pad = pad
        self._ignore_border = ignore_border
        self._mode = mode

    def output(self, x):
        raise NotImplementedError

    def set_input_shape(self, shape):
        raise NotImplementedError

    @staticmethod
    def _get_size_output(input_size, pool_size, stride, pad, ignore_border):

        # extract from Lasagne Library
        if ignore_border:

            return ((input_size + 2 * pad - pool_size + 1) + stride - 1) // stride
        else:

            assert pad == 0

            if stride >= pool_size:
                return (input_size + stride - 1) // stride
            else:
                return max(0, (input_size - pool_size + stride - 1) // stride) + 1


class Pool1D(PoolBase):
    """ Pool 1D Layer.
    """

    def __init__(self, pool_size, mode='max', **kwargs):
        super(Pool1D, self).__init__(pool_size, mode=mode, **kwargs)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        output_shape = list(shape)

        output_shape[-1] = self._get_size_output(shape[-1],
                                             self._pool_size,
                                             self._stride,
                                             self._pad,
                                             self._ignore_border)

        self._output_shape = tuple(output_shape)

    def output(self, x):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation
        """
        return pool_2d(T.shape_padright(x, 1),
                       ds=self._pool_size,
                       st=self._stride,
                       ignore_border=self._ignore_border,
                       padding=self._pad,
                       mode=self._mode)[:, :, :, 0]


class Pool2D(PoolBase):
    """ Pool 2D Layer.
    """

    def __init__(self, pool_size, stride=(1, 1), pad=(0, 0), mode='max', **kwargs):
        super(Pool2D, self).__init__(pool_size, mode=mode, stride=stride, pad=pad, **kwargs)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        output_shape = list(shape)

        output_shape[2] = self._get_size_output(input_size=shape[2],
                                                pool_size=self._pool_size[0],
                                                stride=self._stride[0],
                                                pad=self._pad[0],
                                                ignore_border=self._ignore_border)

        output_shape[3] = self._get_size_output(input_size=shape[3],
                                                pool_size=self._pool_size[1],
                                                stride=self._stride[1],
                                                pad=self._pad[1],
                                                ignore_border=self._ignore_border)

        self._output_shape = tuple(output_shape)

    def output(self, x):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation
        """
        return pool_2d(x,
                       ds=self._pool_size,
                       st=self._stride,
                       ignore_border=self._ignore_border,
                       padding=self._pad,
                       mode=self._mode)


class MaxPool1D(Pool1D):
    """ Max Pool 1D Layer.
    """

    def __init__(self, pool_size, **kwargs):
        super(MaxPool1D, self).__init__(pool_size=pool_size, mode='max', **kwargs)


class MaxPool2D(Pool2D):
    """ Max Pool 2D Layer.
    """

    def __init__(self, pool_size, **kwargs):
        super(MaxPool2D, self).__init__(pool_size=pool_size, mode='max', **kwargs)
