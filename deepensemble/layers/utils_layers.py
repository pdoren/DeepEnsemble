from .layer import Layer

import theano.tensor as T
import numpy as np
from theano import config


__all__ = ['Mask']


class Mask(Layer):

    def __init__(self, input_shape=None, ratio=0.9, seed=13):

        self.__ratio = ratio
        self.__seed = seed
        if input_shape is None:
            output_shape = None
            self.__index = None
        else:
            output_shape = self.__get_output_shape(input_shape, ratio)
            self.__index = self.__generate_index(input_shape, output_shape, seed)

        super(Mask, self).__init__(input_shape=input_shape, output_shape=output_shape, exclude_params=True)

    @staticmethod
    def __get_output_shape(input_shape, ratio):
        output_shape = []
        for i in input_shape:
            d = None
            if i is not None:
                d = int(i * ratio)
            output_shape.append(d)
        return tuple(output_shape)

    @staticmethod
    def __generate_index(input_shape, output_shape, seed):
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

    def output(self, x):
        dim_index = np.squeeze(self.__index).ndim
        if dim_index == 1:
            return x[:, np.squeeze(self.__index)]
        elif dim_index == 2:
            return x[:, np.squeeze(self.__index[:, 0]), np.squeeze(self.__index[:, 1])]
        else:
            raise ValueError('Problem with dimension index mask')
