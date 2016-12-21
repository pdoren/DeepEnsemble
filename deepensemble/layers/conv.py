import theano.tensor as T

from .layer import Layer
from ..utils.utils_functions import ActivationFunctions

__all__ = ['Convolution1D', 'Convolution2D']


class ConvolutionBase(Layer):
    """ Convolution Layer Class Base-

    Parameters
    ----------
    num_filters : int
        Number of filters.

    filter_size : int or tuple[]
        The tuple has the filter size.

    input_shape :  tuple[]
        The tuple has the batch size, num input feature maps and input data size.

    stride : int

    pad : int

    untie_biases : bool

    filter_flip : bool

    non_linearity : callable
    """
    def __init__(self, num_filters, filter_size, input_shape=None, stride=1, pad=0, untie_biases=False,
                 filter_flip=True, non_linearity=ActivationFunctions.linear):

        self._num_filters = num_filters
        self._filter_size = filter_size
        self._stride = stride
        self._pad = pad
        self._untie_biases = untie_biases
        self._filter_flip = filter_flip

        if input_shape is None:

            self._batch_size = None
            self._num_feature_maps = None
            output_shape = None

        else:

            self._batch_size = input_shape[0]
            self._num_feature_maps = input_shape[1]

            output_shape = ((self._batch_size, self._num_filters) +
                            tuple(self._get_size_output(_input, _filter, s, p)
                                  for _input, _filter, s, p
                                  in zip(input_shape[2:], self._filter_size,
                                         self._stride, pad)))

        super(ConvolutionBase, self).__init__(input_shape=input_shape,
                                              output_shape=output_shape,
                                              non_linearity=non_linearity)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        self._output_shape = list(shape)

        self._batch_size = self._input_shape[0]
        self._num_feature_maps = self._input_shape[1]

        output_shape = ((self._batch_size, self._num_filters) +
                        tuple(self._get_size_output(_input, _filter, s, p)
                              for _input, _filter, s, p
                              in zip(self._input_shape[2:], self._filter_size,
                                     self._stride, self._pad)))

        self._output_shape = tuple(output_shape)

    @staticmethod
    def _get_size_output(input_size, filter_size, stride, pad):
        """ Gets size output.

        Parameters
        ----------
        input_size : int
            Size of input layer.

        filter_size : int
            Size of filter (used in convolution).

        stride : int

        pad : int

        Returns
        -------
        int
            Returns size of output.
        """
        # extract from Lasagne Library
        if input_size is None:
            return None
        if pad == 'valid':
            output_size = input_size - filter_size + 1
        elif pad == 'full':
            output_size = input_size + filter_size - 1
        elif pad == 'same':
            output_size = input_size
        elif isinstance(pad, int):
            output_size = input_size + 2 * pad - filter_size + 1
        else:
            raise ValueError('Invalid pad: %s' % pad)

        output_size = (output_size + stride - 1) // stride

        return output_size

    def get_dim_conv(self):
        """ Gets dimension convolution.

        Returns
        -------
        int
            Returns dimension of convolution.
        """
        return len(self.get_input_shape()) - 2

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        return (self._num_filters, self._num_feature_maps) + self._filter_size

    def get_shape_b(self):
        """ Gets shape bias of layer.

        Returns
        -------
        int
            Returns number of filters.
        """
        return self._num_filters,

    def output(self, x, prob=True):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        prob : bool
            No used.

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
            activation = convolution + self._b.dimshuffle(('x', 0) + ('x',) * self.get_dim_conv())

        return self._non_linearity(activation)

    def convolution(self, x):
        """ Compute the convolution.
        """
        raise NotImplementedError


def conv1d_mc0(_input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,), filter_flip=True):
    """ Generate convolution 1D using conv2d with width == 1.

    Parameters
    ----------
    _input : theano.tensor
        Input layer.

    filters : theano.tensor
        Filters.

    image_shape : tuple[]
        Shape of image or array with 1D signals.

    filter_shape : tuple[]
        Shape of filters.

    border_mode : tuple[] or int
        Border mode.

    subsample
        Subsample.

    filter_flip
        Filter flip.

    Returns
    -------
    theano.tensor
        Returns convolution 1D.
    """
    # extract from Lasagne Library
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
    """ Convolution 1D Layer.
    """

    def __init__(self, num_filters, filter_size, input_shape=None, stride=1, pad=0, untie_biases=False,
                 filter_flip=True, non_linearity=ActivationFunctions.linear):
        super(Convolution1D, self).__init__(num_filters=num_filters, filter_size=filter_size, input_shape=input_shape,
                                            stride=stride, pad=pad, untie_biases=untie_biases,
                                            filter_flip=filter_flip, non_linearity=non_linearity)

    def convolution(self, _input):
        """ Convolution 1D Function.

        Parameters
        ----------
        _input : theano.tensor
            Input sample.

        Returns
        -------
        theano.tensor
            Returns convolution 1D.
        """
        border_mode = 'half' if self._pad == 'same' else self._pad
        return conv1d_mc0(_input, self._W,
                          self._input_shape, self.get_shape_W(),
                          subsample=self._stride,
                          border_mode=border_mode,
                          # filter_flip=self._filter_flip
                          )


class Convolution2D(ConvolutionBase):
    """ Convolution 2D Layer.
    """

    def __init__(self, num_filters, filter_size, input_shape=None, stride=(1, 1), pad=(0, 0),
                 untie_biases=False, filter_flip=True, non_linearity=ActivationFunctions.linear):
        super(Convolution2D, self).__init__(num_filters=num_filters, filter_size=filter_size, input_shape=input_shape,
                                            stride=stride, pad=pad, untie_biases=untie_biases,
                                            filter_flip=filter_flip, non_linearity=non_linearity)

    def convolution(self, _input, _conv=T.nnet.conv2d):
        """ Convolution 2D Function.

        Parameters
        ----------
        _input : theano.tensor
            Input sample.

        _conv : theano.Op
            Convolution function.

        Returns
        -------
        theano.tensor
            Returns convolution 1D.
        """
        border_mode = 'half' if self._pad == 'same' else self._pad
        return _conv(_input, self._W,
                     self._input_shape, self.get_shape_W(),
                     subsample=self._stride,
                     border_mode=border_mode,
                     filter_flip=self._filter_flip
                     )
