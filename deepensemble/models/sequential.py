import theano.tensor as T
from theano import function
from .model import Model
from ..utils import *


class Sequential(Model):
    """ This model is a sequence of layers where all elements is interconnected.

    Attributes
    ----------
    layers : list
        List of layers.

    reg_L2 : float
        Ratio of L2 regulation.

    reg_L1 : float
        Ratio of L1 regulation.

    Parameters
    ----------
    name: str
        Name of model.

    type_model: str
        Type of MLP model: classifier or regressor.

    target_labels: list or numpy.array
        Target labels.
    """
    def __init__(self, name, type_model="regressor", target_labels=[]):
        super(Sequential, self).__init__(target_labels=target_labels, type_model=type_model, name=name)
        self.layers = []
        self.reg_L2 = 0.0
        self.reg_L1 = 0.0

    def add_layer(self, new_layer):
        """ Adds new layer.

        Parameters
        ----------
        new_layer : Layer
            New layer.
        """
        n = len(self.layers)
        if n <= 0:
            self.n_input = new_layer.n_input
        else:
            new_layer.n_input = self.layers[n - 1].n_output

        self.layers.append(new_layer)
        self.n_output = new_layer.n_output
        new_layer.initialize_parameters()
        self.params += new_layer.params

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
                for layer in self.layers:
                    _input = layer.output(_input)
                self._output = _input

            return self._output
        else:
            for layer in self.layers:
                _input = layer.output(_input)
            self._output = _input

            return _input

    def reset(self):
        """ Reset parameters
        """
        super(Sequential, self).reset()
        self.params = []
        for layer in self.layers:
            layer.initialize_parameters()
            self.params += layer.params

    def compile(self, fast=True):
        """ Prepare training.

        Parameters
        ----------
        fast : bool
            Compiling cost and regularization items without separating them.
        """
        super(Sequential, self).compile()
        cost = self.get_cost_functions()
        score = self.get_score_functions()

        result = [cost, score]
        if not fast:
            result += self.cost_function_list
            if self.reg_function_list is not None:
                result += self.reg_function_list
            result += self.score_function_list

        updates = self.get_update_function(cost)

        end = T.lscalar('end')
        start = T.lscalar('start')
        r = T.fscalar('r')
        givens = {
            self.model_input: self.share_data_input_train[start:end],
            self.model_target: self.share_data_target_train[start:end],
            self.batch_reg_ratio: r
        }

        self.minibatch_train_eval = function(inputs=[start, end, r], outputs=result, updates=updates,
                                             givens=givens, on_unused_input='ignore')
        self.minibatch_test_eval = function(inputs=[start, end, r], outputs=result,
                                            givens=givens, on_unused_input='ignore')

    def fit(self, _input, _target, max_epoch=100, validation_jump=5, batch_size=32,
            early_stop_th=4, minibatch=True):
        """ Function for training sequential model.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input training samples.

        _target : theano.tensor.matrix
            Target training samples.

        max_epoch : int, 100 by default
            Number of epoch for training.

        validation_jump : int, 5  by default
            Number of times until doing validation jump.

        batch_size : int, 32 by default
            Size of batch.

        early_stop_th : int, 4 by default

        minibatch : bool
            Flag for indicate training with minibatch or not.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        if self.type_model is "classifier":
            metric_model = ClassifierMetrics(self)
        else:
            metric_model = RegressionMetrics(self)

        self.prepare_data(_input, _target)

        for _ in Logger().progressbar_training(max_epoch, self):

            if minibatch:  # Train minibatches
                t_data = self.minibatch_eval(n_input=len(_input), batch_size=batch_size, train=True)
            else:
                t_data = self.minibatch_eval(n_input=len(_input), batch_size=len(_input), train=True)

            metric_model.add_point_train_cost(t_data[0])
            metric_model.add_point_train_score(t_data[1])

        return metric_model
