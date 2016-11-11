import numpy as np

from .layer import Layer
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = ['MaskLayer', 'NoiseLayer']


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

        super(MaskLayer, self).__init__(input_shape=input_shape, output_shape=output_shape, exclude_params=True)

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

    def output(self, _input):
        """ Gets output of model

        Parameters
        ----------
        _input : theano.tensor or numpy.array
            Input sample.

        Returns
        -------

        """
        dim_index = np.squeeze(self.__index).ndim
        if dim_index == 1:
            return _input[:, np.squeeze(self.__index)]
        elif dim_index == 2:
            return _input[:, np.squeeze(self.__index[:, 0]), np.squeeze(self.__index[:, 1])]
        else:
            raise ValueError('Problem with dimension index mask')

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

        super(NoiseLayer, self).__init__(input_shape=input_shape, output_shape=input_shape, exclude_params=True)

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple[]
            Shape of input.
        """
        self._input_shape = shape
        self._output_shape = shape

    def output(self, _input):
        """ Gets output of layer.

        Parameters
        ----------
        _input : theano.tensor or numpy.array
            Input sample.

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
