import theano.tensor as T
from theano import function
import numpy as np
from collections import OrderedDict
from ..combiner.averagecombiner import AverageCombiner
from ..models.model import Model
from ..utils import *

__all__ = ['EnsembleModel']


class EnsembleModel(Model):
    """ Base class Ensemble Model.

    Attributes
    ----------
    combiner : AverageCombiner
        The class combiner allows to mix the models outputs.

    list_models_ensemble : list
        List of the ensemble's models.

    Parameters
    ----------
    name : str, "ensemble" by default
        Ensemble's name.
    """

    def __init__(self, name="ensemble"):
        super(EnsembleModel, self).__init__(target_labels=[], type_model="regressor", name=name, n_input=0, n_output=0)
        self.combiner = None
        self.list_models_ensemble = []
        self.list_cost_ensemble = []

    def set_combiner(self, combiner):
        """ Setter combiner.

        Parameters
        ----------
        combiner : ModelCombiner
            Object ModelCombiner for combining model outputs in ensemble.

        Raises
        ------
        ValueError
            If the combiner method is not same type (regressor or classifier).
        """
        if combiner.type_model is self.type_model:
            self.combiner = combiner
        else:
            raise ValueError("Combiner method must be same type, in this case %s." % self.type_model)

    def append_model(self, new_model):
        """ Add model to ensemble.

        Parameters
        ----------
        new_model : Model
            Model.

        Raises
        ------
        If the model is the different type of the current list the models, it is generated an error.
        """
        if len(self.list_models_ensemble) <= 0:
            # copy data model
            self.n_input = new_model.n_input
            self.n_output = new_model.n_output
            self.type_model = new_model.type_model
            self.target_labels = new_model.target_labels

            self.list_models_ensemble.append(new_model)
        elif self.list_models_ensemble[0] == new_model:
            self.list_models_ensemble.append(new_model)
        else:
            str_error = ''
            if self.list_models_ensemble[0].model.n_input != new_model.n_input:
                str_error += 'different input, '
            if self.list_models_ensemble[0].model.n_output != new_model.n_output:
                str_error += 'different output, '
            if self.list_models_ensemble[0].model.type_model is not new_model.type_model:
                str_error += 'different type learner, '
            if self.list_models_ensemble[0].model.target_labels is not new_model.target_labels:
                str_error += 'different target labels, '

            raise ValueError('Incorrect Learner: ' + str_error[0:-2] + '.')

    def get_num_models(self):
        """ Get number of the Ensemble's models

        Returns
        -------
        int
            Returns current number of models in the Ensemble.
        """
        return len(self.list_models_ensemble)

    def reset(self):
        """ Reset parameters of the ensemble's models.
        """
        super(EnsembleModel, self).reset()
        for model in self.list_models_ensemble:
            model.reset()

    def output(self, _input):
        """ Output of ensemble model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Returns of combiner the outputs of the different the ensemble's models.
        """
        if _input == self.model_input:
            if self._output is None:
                self._output = self.combiner.output(self, _input)
            return self._output
        else:
            return self.combiner.output(self, _input)

    def predict(self, _input):
        """ Compute the prediction of model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Return the prediction of model.
        """
        return self.combiner.predict(self, _input)

    def compile(self, fast=True, **kwargs):
        """ Compile ensemble's models.

        Parameters
        ----------
        fast : bool
            Compiling cost and regularization items without separating them.

        kwargs
            Compilers parameters of models.

        Raises
        ------
        ValueError
            If the combiner method not exists.
        """
        if self.combiner is None:
            raise ValueError("Not exists combiner method for %s." % self.name)

        super(EnsembleModel, self).compile()

        if len(self.list_cost_ensemble) > 0:
            for fun_cost, params_cost in self.list_cost_ensemble:
                for i, model in enumerate(self.list_models_ensemble):
                    model.append_cost(fun_cost=fun_cost, ensemble=self, **params_cost)

        cost = self.get_cost_functions()
        score = self.get_score_functions()
        sub_result = []
        if not fast:  # compute all the scores and costs of the models in ensemble
            for model in self.list_models_ensemble:
                model.set_default_score()
                cost_model = model.get_cost_functions()
                score_model = model.get_score_functions()
                cost += cost_model
                sub_result += [cost_model, score_model]
            result = [cost / self.get_num_models(), score] + sub_result
        else:  # compute only cost and score of ensemble
            for model in self.list_models_ensemble:
                cost_model = model.get_cost_functions()
                cost += cost_model
            result = [cost / self.get_num_models(), score]

        update_combiner = self.combiner.update_parameters(self, self.model_input, self.model_target)
        updates = OrderedDict()
        if update_combiner is not None:
            updates = update_combiner

        for model in self.list_models_ensemble:
            cost_model = model.get_cost_functions()
            update_model = model.get_update_function(cost_model)
            for key in update_model.keys():
                updates[key] = update_model[key]

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

    def fit(self, _input, _target, max_epoch, validation_jump, minibatch=True, batch_size=32, **kwargs):
        """ Training ensemble.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Training Input sample.

        _target : theano.tensor.matrix
            Training Target sample.

        max_epoch : int
            Number of epoch for training.

        validation_jump : int
            Number of times until doing validation jump.

        minibatch : bool
            Flag for indicate training with minibatch or not.

        batch_size: int
            Size of batch.

        kwargs
            Other parameters.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        metric_models = []
        if self.type_model is "classifier":
            metric_ensemble = EnsembleClassifierMetrics(self)
            for model in self.list_models_ensemble:
                metric_models.append(ClassifierMetrics(model))
        else:
            metric_ensemble = EnsembleRegressionMetrics(self)
            for model in self.list_models_ensemble:
                metric_models.append(RegressionMetrics(model))

        self.prepare_data(_input, _target)

        for _ in Logger().progressbar_training(max_epoch, self):

            if minibatch:  # Train minibatches
                t_data = self.minibatch_eval(n_input=len(_input), batch_size=batch_size, train=True)
            else:
                t_data = self.minibatch_eval(n_input=len(_input), batch_size=len(_input), train=True)

            metric_ensemble.add_point_train_cost(t_data[0])
            metric_ensemble.add_point_train_score(t_data[1])

            if len(t_data) > 2:
                for i, model in enumerate(self.list_models_ensemble):
                    ind = 2 * (i + 1)
                    metric_models[i].add_point_train_cost(t_data[ind])
                    metric_models[i].add_point_train_score(t_data[ind + 1])

        for metric in metric_models:
            metric_ensemble.append_metric(metric)

        return metric_ensemble

    def add_cost_ensemble(self, fun_cost, **kwargs):
        """ Adds cost function for each models in Ensemble.

        Parameters
        ----------
        fun_cost : theano.function
            Cost function.

        kwargs
            Other parameters.
        """
        self.list_cost_ensemble.append((fun_cost, kwargs))
