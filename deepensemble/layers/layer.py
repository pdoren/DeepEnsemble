from theano import shared, config
import theano.tensor as T
import numpy as np

__all__ = ['Layer']


class Layer(object):
    """ Base class layers.

    Attributes
    ----------
    __input_shape : tuple
        Input shape.

    __output_shape : tuple
        Output shape.

    _non_linearity : theano.Op
        Activation functions.

    _W : theano.shared
        Weights.

    _b : theano.shared
        Bias.

    _params : list
        Parameters of model: [W, b].

    Parameters
    ----------

    """
    def __init__(self, input_shape=None, output_shape=None, non_linearity=None, exclude_params=False):
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self._params = []
        self._W = None
        self._b = None
        self._non_linearity = non_linearity
        self.__exclude_params = exclude_params

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
        raise NotImplementedError

    def exclude_params(self):
        return self.__exclude_params

    def get_input_shape(self):
        return self.__input_shape

    def set_input_shape(self, shape):
        self.__input_shape = shape

    def get_output_shape(self):
        return self.__output_shape

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        return (0,)

    def get_shape_b(self):
        """ Gets shape bias of layer.
        """
        return (0,)

    def initialize_parameters(self):
        """ Initialize neurons params of layers
        """
        if not self.__exclude_params:
            if self._W is None:
                self._W = shared(np.zeros(shape=self.get_shape_W(), dtype=config.floatX), name='W', borrow=True)
            if self._b is None:
                self._b = shared(np.zeros(shape=self.get_shape_b(), dtype=config.floatX), name='b', borrow=True)

            wb = np.sqrt(6.0 / (self.get_fan_in() + self.get_fan_out()))  # W bound

            W = np.array(np.random.uniform(low=-wb, high=wb, size=self.get_shape_W()), dtype=config.floatX)
            if self._non_linearity == T.nnet.sigmoid:
                W *= 4

            self._W.set_value(W)
            self._b.set_value(np.zeros(shape=self.get_shape_b(), dtype=config.floatX))

            self._params = [self._W, self._b]

    def get_fan_in(self):
        """ Getter of input dimension.

        Returns
        -------
        int
            Returns input dimension of layer.
        """
        return int(np.prod(self.__input_shape))

    def get_fan_out(self):
        """ Getter of output dimension.

        Returns
        -------
        int
            Returns output dimension of layer.
        """
        return int(np.prod(self.__output_shape))

    def get_parameters(self):
        """ Getter of parameters of layer.

        Returns
        -------
        list
            Returns one list with all parameters of layer.
        """
        return self._params

    def get_W(self):
        """ Get weights of Layer.

        Returns
        -------
        theano.tensor.matrix
            Returns weights of layer.
        """
        return self._W

    def get_b(self):
        """ Get bias of Layer.

        Returns
        -------
        theano.tensor.matrix
            Returns bias of layer.
        """
        return self._b
