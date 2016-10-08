import theano.tensor.nnet.conv as conv
import theano.tensor as T
import numpy as np
from .layer import Layer
from ..utils.utils_functions import ActivationFunctions

__all__ = ['Convolution1D', 'Convolution2D']


class ConvolutionBase(Layer):
    def __init__(self, filter_shape, input_shape=None, stride=1, pad=0, untie_biases=False, filter_flip=True,
                 non_linearity=ActivationFunctions.linear):
        """

        Parameters
        ----------
        filter_shape : tuple
            The tuple has the number of filters, num input feature maps and filter size.

        input_shape :  tuple
            The tuple has the batch size, num input feature maps and input data size.

        stride : int

        pad : int

        untie_biases : bool

        filter_flip : bool

        non_linearity : callable
        """

        fan_out = int()

        self._filter_shape = filter_shape
        self._stride = stride
        self._pad = pad
        self._untie_biases = untie_biases
        self._filter_flip = filter_flip
        self._dim_conv = len(input_shape) - 2

        super(ConvolutionBase, self).__init__(input_shape=input_shape,
                                              output_shape=(filter_shape[0],) + np.prod(filter_shape[2:]),
                                              non_linearity=non_linearity)

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        return self._filter_shape

    def get_shape_b(self):
        """ Gets shape bias of layer.
        """
        return self._filter_shape[0],

    def output(self, x):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers.
        """
        convolution = self.convolution(x)

        if self._b is None:
            activation = convolution
        elif self._untie_biases:
            activation = convolution + T.shape_padleft(self._b, 1)
        else:
            activation = convolution + self._b.dimshuffle(('x', 0) + ('x',) * self._dim_conv)

        return self._non_linearity(activation)

    def convolution(self, x):
        """ Compute the convolution.
        """
        raise NotImplementedError


def conv1d_mc0(_input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,), filter_flip=True):
    """
    using conv2d with width == 1
    """
    if image_shape is None:
        image_shape_mc0 = None
    else:
        # (b, c, i0) to (b, c, 1, i0)
        image_shape_mc0 = (image_shape[0], image_shape[1], 1, image_shape[2])

    if filter_shape is None:
        filter_shape_mc0 = None
    else:
        filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1,
                            filter_shape[2])

    if isinstance(border_mode, tuple):
        (border_mode,) = border_mode
    if isinstance(border_mode, int):
        border_mode = (0, border_mode)

    input_mc0 = _input.dimshuffle(0, 1, 'x', 2)
    filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)

    conved = T.nnet.conv2d(
        input_mc0, filters_mc0, image_shape_mc0, filter_shape_mc0,
        subsample=(1, subsample[0]), border_mode=border_mode,
        filter_flip=filter_flip)
    return conved[:, :, 0, :]  # drop the unused dimension


class Convolution1D(ConvolutionBase):
    def __init__(self, filter_shape, input_shape, stride=1, pad=0, untie_biases=False, filter_flip=True,
                 non_linearity=ActivationFunctions.linear):
        super(Convolution1D, self).__init__(filter_shape=filter_shape, input_shape=input_shape,
                                            stride=stride, pad=pad, untie_biases=untie_biases,
                                            filter_flip=filter_flip, non_linearity=non_linearity)

    def convolution(self, x):
        border_mode = 'half' if self._pad == 'same' else self._pad
        return conv1d_mc0(x, self._W,
                          self._input_shape, self.get_shape_W(),
                          subsample=self._stride,
                          border_mode=border_mode,
                          filter_flip=self._filter_flip
                          )


class Convolution2D(ConvolutionBase):
    def __init__(self, filter_shape, input_shape, stride=(1, 1), pad=0, untie_biases=False, filter_flip=True,
                 non_linearity=ActivationFunctions.linear):
        super(Convolution2D, self).__init__(filter_shape=filter_shape, input_shape=input_shape,
                                            stride=stride, pad=pad, untie_biases=untie_biases,
                                            filter_flip=filter_flip, non_linearity=non_linearity)

    def convolution(self, x):
        border_mode = 'half' if self._pad == 'same' else self._pad
        return conv.conv2d(x, self._W,
                           self._input_shape, self.get_shape_W(),
                           subsample=self._stride,
                           border_mode=border_mode,
                           filter_flip=self._filter_flip
                           )
