from .model import Model
from ..metrics import *
from ..utils import translate_output

__all__ = ['Sequential']


class Sequential(Model):
    """ This model is a sequence of layers where all elements is interconnected.

    Attributes
    ----------
    __layers : list
        List of layers.

    Parameters
    ----------
    name: str
        Name of model.

    type_model: str
        Type of MLP model: classifier or regressor.
    """

    def __init__(self, name, type_model="regressor", target_labels=None):
        if type_model == "regressor":
            target_labels = []
        super(Sequential, self).__init__(target_labels=target_labels, type_model=type_model, name=name)
        self.__layers = []

    def get_layers(self):
        """ Get list of layers.

        Returns
        -------
        list
            Returns a list of layers of this model.
        """
        return self.__layers

    def get_new_metric(self):
        """ Get metric of respective model.

        Returns
        -------
        BaseMetrics
            Returns a metric that will depend on type of model.
        """
        if self.is_classifier():
            return ClassifierMetrics(self)
        else:
            return RegressionMetrics(self)

    def add_layer(self, new_layer):
        """ Adds new layer.

        Parameters
        ----------
        new_layer : Layer
            New layer.
        """
        n = len(self.__layers)
        if n <= 0:
            self.set_input_shape(shape=new_layer.get_input_shape())
            self._define_input()
        else:
            new_layer.set_input_shape(self.__layers[n - 1].get_output_shape())

        self.__layers.append(new_layer)
        self.set_output_shape(shape=(new_layer.get_output_shape()))

        new_layer.initialize_parameters()
        self._params += new_layer.get_parameters()

    def output(self, _input, prob=True):
        """ Output of sequential model.

        Parameters
        ----------
        _input: theano.tensor.matrix or numpy.array
            Input sample.

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Returns the output sequential model.
        """
        if _input == self._model_input:

            _type = 'prob' if prob else 'crisp'

            if self._output[_type]['changed']:
                for layer in self.__layers:
                    _input = layer.output(_input)

                self._output[_type]['result'] = _input

                if _type == 'crisp' and self.is_classifier():
                    self._output[_type]['result'] = translate_output(self._output[_type]['result'],
                                                                     self.get_fan_out(),
                                                                     self.is_binary_classification())
                self._output[_type]['changed'] = False

            return self._output[_type]['result']

        else:

            for layer in self.__layers:
                _input = layer.output(_input)
            _output = _input

            if not prob and self.is_classifier():
                self._output = translate_output(_output, self.get_fan_out(), self.is_binary_classification())

            return _output

    def reset(self):
        """ Reset parameters
        """
        super(Sequential, self).reset()
        self._params = []
        for layer in self.__layers:
            layer.initialize_parameters()
            self._params += layer.get_parameters()

    def _compile(self, fast=True, **kwargs):
        """ Prepare training.

        Parameters
        ----------
        fast : bool
            Compiling cost and regularization items without separating them.
        """
        self._define_output()

        cost = self.get_cost()

        return cost, self.get_update_function(cost), [], []
