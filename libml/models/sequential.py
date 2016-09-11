from theano import function
import numpy as np
from .model import Model
from libml.utils.utils_classifiers import *
from libml.utils.metrics.classifiermetrics import ClassifierMetrics
from libml.utils.metrics.regressionmetrics import RegressionMetrics


class Sequential(Model):
    def __init__(self, target_labels, type_model, name):
        """ Base class for a generic model. This model is a sequence of layers.

        Parameters
        ----------
        target_labels: list or numpy.array
            Target labels.

        type_model: str
            Type of MLP model: classifier or regressor.

        name: str
            Name of model.
        """
        super(Sequential, self).__init__(target_labels=target_labels, type_model=type_model, name=name)
        self.layers = []
        self.fun_train = None
        self.fun_test = None
        self.reg_L2 = 0.0
        self.reg_L1 = 0.0

    def add_layer(self, new_layer):
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
        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
        Returns the output sequential model.

        """
        for layer in self.layers:
            _input = layer.output(_input)
        return _input

    def reset(self):
        """ Reset parameters
        """
        self.params = []
        for layer in self.layers:
            layer.initialize_parameters()
            self.params += layer.params

    def compile(self):
        """ Prepare training.
        """
        super(Sequential, self).compile()

        if self.reg_function is not None:
            self.cost_function += self.reg_function

        self.fun_train = function([self.model_input, self.model_target, self.batch_reg_ratio],
                                  [self.cost_function, self.score],
                                  updates=self.updates, on_unused_input='ignore')
        self.fun_test = function([self.model_input, self.model_target, self.batch_reg_ratio],
                                 [self.cost_function, self.score],
                                 on_unused_input='ignore')

    def minibatch_eval(self, _input, _target, batch_size=32, train=True):
        """ Evaluate cost and score in mini batch.

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        _target: theano.tensor.matrix
            Target sample.

        batch_size: int
            Size of batch.

        train: bool
            Flag for knowing if the evaluation of batch is for training or testing.

        Returns
        -------
        tuple
            Returns evaluation cost and score in mini batch.
        """
        averaged_cost = 0.0
        averaged_score = 0.0
        N = len(_input)
        NN = 0
        for (start, end) in zip(range(0, len(_input), batch_size), range(batch_size, len(_input), batch_size)):
            r = (end - start) / N
            NN += 1
            if train:
                cost, score = self.fun_train(_input[start:end], _target[start:end], r)
                averaged_cost += cost
                averaged_score += score
            else:
                cost, score = self.fun_test(_input[start:end], _target[start:end], r)
                averaged_cost += cost
                averaged_score += score
        return averaged_cost / NN, averaged_score / NN

    def fit(self, _input, _target, max_epoch=100, validation_jump=5, batch_size=32,
            early_stop_th=4, verbose=False):
        """

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

        verbose : bool, False by default
            Flag for show training information.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.

        """
        if self.type_model is "classifier":
            metrics = ClassifierMetrics(self)
        else:
            metrics = RegressionMetrics(self)

        target_train = _target
        input_train = _input
        if self.type_model is 'classifier':
            target_train = translate_target(_target=_target, n_classes=self.n_output, target_labels=self.target_labels)

        for epoch in range(0, max_epoch):
            # Present mini-batches in different order
            rand_perm = np.random.permutation(len(target_train))
            input_train = input_train[rand_perm]
            target_train = target_train[rand_perm]

            # Train minibatches
            train_cost, train_score = self.minibatch_eval(_input=input_train, _target=target_train,
                                                          batch_size=batch_size, train=True)
            metrics.append_train_cost(train_cost)
            metrics.append_train_score(train_score)

            if verbose:
                print("epoch %i" % epoch)

        return metrics
