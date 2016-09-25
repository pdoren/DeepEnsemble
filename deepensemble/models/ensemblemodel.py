from collections import OrderedDict

import theano.tensor as T
from theano import function

from .model import Model
from ..metrics import *

__all__ = ['EnsembleModel']


class EnsembleModel(Model):
    """ Base class Ensemble Model.

    Attributes
    ----------
    __combiner : AverageCombiner
        The class combiner allows to mix the models outputs.

    __list_models_ensemble : list[Model]
        List of the ensemble's models.

    Parameters
    ----------
    name : str, "ensemble" by default
        Ensemble's name.
    """

    def __init__(self, name="ensemble"):
        super(EnsembleModel, self).__init__(target_labels=[], type_model="regressor", name=name, n_input=0, n_output=0)
        self.__combiner = None
        self.__list_models_ensemble = []
        self.__list_cost_ensemble = []
        self._params.append(0)  # the first element is reserved for combiner parameters

    def get_models(self):
        return self.__list_models_ensemble

    def get_new_metric(self):
        if self._type_model == "classifier":
            return EnsembleClassifierMetrics(self)
        else:
            return EnsembleRegressionMetrics(self)

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
        if combiner.get_type_model() is self._type_model:
            self.__combiner = combiner
            self._params[0] = combiner.get_params()
        else:
            raise ValueError("Combiner method must be same type, in this case %s." % self._type_model)

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
        if len(self.__list_models_ensemble) <= 0:
            # copy data model
            self._n_input = new_model._n_input
            self._n_output = new_model._n_output
            self._type_model = new_model._type_model
            self._target_labels = new_model._target_labels

            self.__list_models_ensemble.append(new_model)
        elif self.__list_models_ensemble[0] == new_model:
            self.__list_models_ensemble.append(new_model)
        else:
            str_error = ''
            if self.__list_models_ensemble[0].model.n_input != new_model._n_input:
                str_error += 'different input, '
            if self.__list_models_ensemble[0].model.n_output != new_model._n_output:
                str_error += 'different output, '
            if self.__list_models_ensemble[0].model.type_model is not new_model._type_model:
                str_error += 'different type learner, '
            if self.__list_models_ensemble[0].model.get_target_labels() is not new_model.get_target_labels():
                str_error += 'different target labels, '

            raise ValueError('Incorrect Learner: ' + str_error[0:-2] + '.')

        self._params += new_model.get_params()

    def get_num_models(self):
        """ Get number of the Ensemble's models

        Returns
        -------
        int
            Returns current number of models in the Ensemble.
        """
        return len(self.__list_models_ensemble)

    def reset(self):
        """ Reset parameters of the ensemble's models.
        """
        super(EnsembleModel, self).reset()
        for model in self.__list_models_ensemble:
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
                self._output = self.__combiner.output(self, _input)
            return self._output
        else:
            return self.__combiner.output(self, _input)

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
        return self.__combiner.predict(self, _input)

    def _compile(self, fast=True, **kwargs):
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
        if self.__combiner is None:
            raise ValueError("Not exists combiner method for %s." % self._name)

        # append const ensemble for each models
        if len(self.__list_cost_ensemble) > 0:
            for fun_cost, params_cost in self.__list_cost_ensemble:
                for i, model in enumerate(self.__list_models_ensemble):
                    model.append_cost(fun_cost=fun_cost, ensemble=self, **params_cost)

        cost = self.get_cost_functions()
        score = self.get_score_functions()
        m = self.get_num_models()
        if not fast:  # compute all the scores and costs of the models in ensemble
            sub_result = []
            for model in self.__list_models_ensemble:
                model.set_default_score()
                cost_model = model.get_cost_functions()
                score_model = model.get_score_functions()
                cost += cost_model
                sub_result += [cost_model, score_model]
            result = [cost / m, score] + sub_result
        else:  # compute only cost and score of ensemble
            for model in self.__list_models_ensemble:
                cost_model = model.get_cost_functions()
                cost += cost_model
            result = [cost / m, score]

        update_combiner = self.__combiner.update_parameters(self, self.model_input, self.model_target)
        updates = OrderedDict()
        if update_combiner is not None:
            updates = update_combiner

        for model in self.__list_models_ensemble:
            cost_model = model.get_cost_functions()
            update_model = model.get_update_function(cost_model)
            for key in update_model.keys():
                updates[key] = update_model[key]

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
                                              givens=givens_train, on_unused_input='ignore')
        self._minibatch_test_eval = function(inputs=[start, end, r], outputs=result,
                                             givens=givens_test, on_unused_input='ignore')

    def add_cost_ensemble(self, fun_cost, **kwargs):
        """ Adds cost function for each models in Ensemble.

        Parameters
        ----------
        fun_cost : theano.function
            Cost function.

        kwargs
            Other parameters.
        """
        self.__list_cost_ensemble.append((fun_cost, kwargs))
