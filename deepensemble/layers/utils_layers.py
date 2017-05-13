from collections import OrderedDict

import numpy as np
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from ..utils.utils_translation import TextTranslation

from .layer import Layer

__all__ = ['MaskLayer', 'NoiseLayer', 'BiasLayer']


class MaskLayer(Layer):
    """ This layer generate a random permutation on index features or inputs of layer.

    Parameters
    ----------
    input_shape : tuple[]
        Tuple input layer.

    ratio : float
        This number is used as ratio

    seed
        Number of used as seed for random number generator (see numpy.random.seed).
    """

    def __init__(self, input_shape=None, ratio=0.9, seed=13):
        self.__ratio = ratio
        self.__seed = seed
        if input_shape is None:
            output_shape = None
            self.__index = None
        else:
            output_shape = self.__get_output_shape(input_shape, ratio)
            self.__index = self.__generate_index(input_shape, output_shape, seed)

        super(MaskLayer, self).__init__(input_shape=input_shape, output_shape=output_shape,
                                        include_b=False, include_w=False)

    @staticmethod
    def __get_output_shape(input_shape, ratio):
        """ Gets the output shape.

        Parameters
        ----------
        input_shape : tuple[]
            Input shape.

        ratio : float
            Ratio that define output shape according to following relation:
                output_shape = ratio * input_shape.

        Returns
        -------
        tuple[]
            Returns output shape.
        """
        output_shape = []
        for i in input_shape:
            d = None
            if i is not None:
                d = int(i * ratio)
            output_shape.append(d)
        return tuple(output_shape)

    @staticmethod
    def __generate_index(input_shape, output_shape, seed):
        """ Generate an array with permutation on index features or inputs of layer.

        Parameters
        ----------
        input_shape : tuple[]
            Shape of input layer.

        output_shape : tuple[]
            Shape of output layer.

        seed
            Number of used as seed for random number generator (see numpy.random.seed).

        See Also
        --------
        numpy.random.seed

        Returns
        -------
        numpy.array
            Return an array with permutation index.
        """
        input_shape = list(input_shape)
        output_shape = list(output_shape)
        del input_shape[input_shape is None]
        del output_shape[output_shape is None]
        np.random.seed(seed)
        return np.array([np.random.permutation(n)[0:output_shape[i]] for i, n in enumerate(input_shape)])

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        self._output_shape = self.__get_output_shape(shape, self.__ratio)
        self.__index = self.__generate_index(self._input_shape, self._output_shape, self.__seed)

    def output(self, _input, prob=True):
        """ Gets output of model

        Parameters
        ----------
        _input : theano.tensor or numpy.array
            Input sample.

        prob : bool
            Flag for changing behavior of some layers.

        Returns
        -------

        """
        dim_index = np.squeeze(self.__index).ndim
        if dim_index == 1:
            return _input[:, np.squeeze(self.__index)]
        elif dim_index == 2:
            return _input[:, np.squeeze(self.__index[:, 0]), np.squeeze(self.__index[:, 1])]
        else:
            raise ValueError(TextTranslation().get_str('Error_6'))


class NoiseLayer(Layer):
    """ This layer added noise.

    Parameters
    ----------
    input_shape : tuple[]
        Tuple input layer.

    seed
        Number of used as seed for random number generator.

    rng : str
        Type of distribution (uniform, binomial, normal).

    seed
        Number of used as seed for random number generator.

    kwargs
        Parameters of distribution.
    """

    def __init__(self, input_shape=None, seed=13, rng='normal', **kwargs):

        srng = RandomStreams(seed=seed)
        self._rng_params = kwargs
        if rng == 'uniform':
            self._rng = srng.uniform
        elif rng == 'binomial':
            self._rng = srng.binomial
        else:
            self._rng = srng.normal

        super(NoiseLayer, self).__init__(input_shape=input_shape, output_shape=input_shape,
                                         include_w=False, include_b=False)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        self._output_shape = shape

    def output(self, _input, prob=True):
        """ Gets output of layer.

        Parameters
        ----------
        _input : theano.tensor or numpy.array
            Input sample.

        prob : bool

        Returns
        -------
        theano.tensor
            Returns input plus noise.
        """
        input_shape = self.get_input_shape()
        if any(s is None for s in input_shape):
            input_shape = _input.shape

        return _input + self._rng(input_shape,
                                  dtype=config.floatX, **self._rng_params)


class BiasLayer(Layer):
    """ This layer added noise.

    Parameters
    ----------
    input_shape : tuple[]
        Tuple input layer.

    seed
        Number of used as seed for random number generator.

    rng : str
        Type of distribution (uniform, binomial, normal).

    seed
        Number of used as seed for random number generator.

    kwargs
        Parameters of distribution.
    """

    # noinspection PyUnusedLocal
    def __init__(self, net, input_shape=None, **kwargs):
        super(BiasLayer, self).__init__(input_shape=input_shape, output_shape=input_shape,
                                        include_w=False, include_b=True)

        self._b[0]['name'] = 'bias'
        self.set_include_b(False)

        net.append_update(self.update, TextTranslation().get_str('Update_BiasLayer'))
        self._updates = OrderedDict()

    def update(self, error):
        self._updates[self.get_b()] = T.mean(error, axis=0, dtype=config.floatX)
        return self._updates

    def set_b(self, b):
        return self.get_b().set_value(b)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        self._output_shape = shape
        self.update_b_shape()

    def output(self, _input, prob=True):
        """ Gets output of layer.

        Parameters
        ----------
        _input : theano.tensor or numpy.array
            Input sample.

        prob : bool
            This flag activates bias.

        Returns
        -------
        theano.tensor
            Returns input plus noise.
        """
        x = _input

        if _input.ndim > 2:
            x = _input.flatten(2)

        return x - self.get_b().dimshuffle('x', 0)
