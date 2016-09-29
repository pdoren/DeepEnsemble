import theano.tensor as T
from theano import function

from .model import Model
from ..metrics import *

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
        if self._type_model == "classifier":
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
            self._n_input = new_layer.get_n_inputs()
        else:
            new_layer.set_n_inputs(self.__layers[n - 1].get_n_outputs())

        self.__layers.append(new_layer)
        self._n_output = new_layer.get_n_outputs()
        new_layer.initialize_parameters()
        self._params += new_layer.get_parameters()

    def output(self, _input):
        """ Output of sequential model.

        Parameters
        ----------
        _input: theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Returns the output sequential model.
        """
        if _input == self.model_input:
            if self._output is None:
                for layer in self.__layers:
                    _input = layer.output(_input)
                self._output = _input

            return self._output
        else:
            for layer in self.__layers:
                _input = layer.output(_input)
            self._output = _input

            return _input

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
        cost = self.get_cost_functions()
        score = self.get_score_functions()

        result = [cost, score]
        if not fast:
            result += self._cost_function_list
            if self._reg_function_list is not None:
                result += self._reg_function_list
            result += self._score_function_list

        updates = self.get_update_function(cost)

        end = T.lscalar('end')
        start = T.lscalar('start')
        r = T.fscalar('r')
        givens_train = {
            self.model_input: self._share_data_input_train[start:end],
            self.model_target: self._share_data_target_train[start:end],
            self.batch_reg_ratio: r
        }

        givens_test = {
            self.model_input: self._share_data_input_test[start:end],
            self.model_target: self._share_data_target_test[start:end],
            self.batch_reg_ratio: r
        }

        self._minibatch_train_eval = function(inputs=[start, end, r], outputs=result, updates=updates,
                                              givens=givens_train, on_unused_input='ignore', allow_input_downcast=True)
        self._minibatch_test_eval = function(inputs=[start, end, r], outputs=result,
                                             givens=givens_test, on_unused_input='ignore', allow_input_downcast=True)
