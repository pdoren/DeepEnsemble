from theano import shared, config
import theano.tensor as T
import numpy as np
from ..utils.serializable import Serializable

__all__ = ['Layer']


# noinspection PyTypeChecker
class Layer(Serializable):
    """ Base class layers.

    Attributes
    ----------
    _input_shape : tuple[]
        Input shape.

    _output_shape : tuple[]
        Output shape.

    _non_linearity : theano.Op
        Activation functions.

    _W : list[dict]
        Weights.

    _b : list[]
        Bias.

    _params : list
        Parameters of model: [W, b].

    Parameters
    ----------

    """

    def __init__(self, input_shape=None, output_shape=None, non_linearity=None, include_w=True, include_b=True):
        super(Layer, self).__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._params = []
        # default W and b
        if include_w:
            self._W = [{'name': 'W', 'value': None, 'shape': None, 'init': include_w, 'include': include_w}]
        else:
            self._W = []

        if include_b:
            self._b = [{'name': 'b', 'value': None, 'shape': None, 'init': include_b, 'include': include_b}]
        else:
            self._b = []

        self.update_W_shape()
        self.update_b_shape()

        self._non_linearity = non_linearity

    def update_W_shape(self):
        for w in self._W:
            w['shape'] = (self.get_fan_in(), self.get_fan_out())

    def update_b_shape(self):
        for b in self._b:
            b['shape'] = (self.get_fan_out(),)

    def set_include_W(self, state=True):
        for w in self._W:
            w['include'] = state

    def set_include_b(self, state=True):
        for b in self._b:
            b['include'] = state

    def output(self, x, prob=True):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation
        """
        raise NotImplementedError

    def get_input_shape(self):
        """ Gets shape of input.

        Returns
        -------
        tuple[]
            Returns shape of input model.
        """
        return self._input_shape

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple
            Shape of input.
        """
        self._input_shape = shape
        self.update_W_shape()

    def get_output_shape(self):
        """ Gets output shape.

        Returns
        -------
        tuple
            Returns output shape.
        """
        return self._output_shape

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        if len(self._W) > 0:
            return self._W[0]['shape']
        else:
            return 0,

    def get_shape_b(self):
        """ Gets shape bias of layer.
        """
        if len(self._b) > 0:
            return self._b[0]['shape']
        else:
            return 0,

    def initialize_parameters(self):
        """ Initialize neurons params of layers
        """
        self._params = []

        for w in self._W:
            if w['init']:
                if w['value'] is None:
                    w['value'] = shared(np.zeros(shape=w['shape'], dtype=config.floatX), name=w['name'], borrow=True)

                w['value'].set_value(self.init_W(w['shape']))

            self._params.append(w)

        for b in self._b:
            if b['init']:
                if b['value'] is None:
                    b['value'] = shared(np.zeros(shape=b['shape'], dtype=config.floatX), name=b['name'], borrow=True)

                b['value'].set_value(np.zeros(shape=b['shape'], dtype=config.floatX))

            self._params.append(b)

    def init_W(self, shape_W):
        """ Initialize Weights.
        """
        wb = np.sqrt(6.0 / (self.get_fan_in() + self.get_fan_out()))  # W bound
        W = np.array(np.random.uniform(low=-wb, high=wb, size=shape_W), dtype=config.floatX)
        if self._non_linearity == T.nnet.sigmoid:
            W *= 4.0

        return W

    def get_fan_in(self):
        """ Getter of input dimension.

        Returns
        -------
        int
            Returns input dimension of layer.
        """
        if self._input_shape is None:
            return 0
        else:
            sn = np.prod(self._input_shape[1:])
            if sn is None:
                return 0
            else:
                return int(sn)

    def get_fan_out(self):
        """ Getter of output dimension.

        Returns
        -------
        int
            Returns output dimension of layer.
        """
        if self._output_shape is None:
            return 0
        else:
            return int(np.prod(self._output_shape[1:]))

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
        return self._W[0]['value']

    def get_b(self):
        """ Get bias of Layer.

        Returns
        -------
        theano.tensor.matrix
            Returns bias of layer.
        """
        return self._b[0]['value']
