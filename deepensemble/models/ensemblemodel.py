from collections import OrderedDict

from sklearn import cross_validation

from .model import Model
from ..metrics import *
from ..utils import Logger

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
        super(EnsembleModel, self).__init__(target_labels=[], type_model="regressor", name=name)
        self.__combiner = None
        self.__list_models_ensemble = []
        self.__list_cost_ensemble = []
        self._params.append(0)  # the first element is reserved for combiner parameters

    def get_combiner(self):
        return self.__combiner

    def get_models(self):
        """ Getter list of ensemble models.

        Returns
        -------
        list
            Returns list of models.
        """
        return self.__list_models_ensemble

    def get_new_metric(self):
        """ Gets metric for this model, function necessary for FactoryMetrics.

        Returns
        -------
        EnsembleMetrics
            Returns ensemble metrics.

        See Also
        --------
        FactoryMetrics
        """
        if self.is_classifier():
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
        if combiner.get_type_model() is self.get_type_model():
            self.__combiner = combiner
            self._params[0] = combiner.get_params()
        else:
            raise ValueError("Combiner method must be same type, in this case %s." % self.__type_model)

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
            self.copy_kind_of_model(new_model)  # copy model
            self.__list_models_ensemble.append(new_model)
        elif self.__list_models_ensemble[0] == new_model:
            self.__list_models_ensemble.append(new_model)
        else:
            str_error = ''
            if self.__list_models_ensemble[0].model.get_input_shape() != new_model.get_input_shape():
                str_error += 'different input, '
            if self.__list_models_ensemble[0].model.get_output_shape() != new_model.get_output_shape():
                str_error += 'different output, '
            if self.__list_models_ensemble[0].model.type_model != new_model.get_type_model():
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

    def output(self, _input, prob=True):
        """ Output of ensemble model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
            Input sample.

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Returns of combiner the outputs of the different the ensemble's models.
        """
        if _input == self.model_input:
            if self._output is None:
                self._output = self.__combiner.output(self, _input, prob)
            return self._output
        else:
            return self.__combiner.output(self, _input, prob)

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

    def _compile(self, fast=True, full_costs=True, full_scores=True, **kwargs):
        """ Compile ensemble's models.

        Parameters
        ----------
        fast : bool
            Compiling cost and regularization items without separating them.

        full_costs : bool
            Flag to active save all data model costs in training.

        full_scores : bool
            Flag to active save all data model scores in training.

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
            for fun_cost, name, params_cost in self.__list_cost_ensemble:
                for i, model in enumerate(self.get_models()):
                    model.append_cost(fun_cost=fun_cost, name=name, ensemble=self, **params_cost)

        cost = []
        extra_results = []
        labels_extra_results = []
        if not fast:  # compute all the scores and costs of the models in ensemble
            for model in self.get_models():
                model.review_is_binary_classifier()  # update if is binary classifier
                cost_model = model.get_cost()
                cost += [cost_model]
                extra_results += [cost_model]
                labels_extra_results += ['Cost']
                labels_extra_results += model.get_labels_costs()
                extra_results += model.get_costs()
                labels_extra_results += model.get_labels_scores()
                extra_results += model.get_scores()
        else:  # compute only cost and score of ensemble
            for model in self.get_models():
                model.review_is_binary_classifier()  # update if is binary classifier
                cost_model = model.get_cost()
                cost += [cost_model]

        cost = sum(cost) / self.get_num_models()

        update_combiner = self.__combiner.update_parameters(self, self.model_input, self.model_target)
        updates = OrderedDict()
        if update_combiner is not None:
            updates = update_combiner

        for model in self.get_models():
            cost_model = model.get_cost()
            update_model = model.get_update_function(cost_model)
            for key in update_model.keys():
                updates[key] = update_model[key]

        return cost, updates, extra_results, labels_extra_results

    def add_cost_ensemble(self, fun_cost, name, **kwargs):
        """ Adds cost function for each models in Ensemble.

        Parameters
        ----------
        fun_cost : theano.function
            Cost function.

        name : str
            This string identify cost function, is useful for plot metrics.

        kwargs
            Other parameters.
        """
        self.__list_cost_ensemble.append((fun_cost, name, kwargs))

    def fit_folds(self, _input, _target, seed=13, **kwargs):
        """ Function for training sequential model.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input training samples.

        _target : theano.tensor.matrix
            Target training samples.

        seed : int
            Seed for random generators.

        kwargs

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        # create a specific metric
        metric_model = FactoryMetrics().get_metric(self)

        nfolds = self.get_num_models()

        folds = list(cross_validation.StratifiedKFold(_target, nfolds, shuffle=True, random_state=seed))
        metric_model = FactoryMetrics().get_metric(self)

        iterator = zip(self.get_models(), folds)

        best_score = 0
        for model, train_index, test_index in Logger().progressbar_training2(iterator, self):
            fold_X_train = _input[train_index]
            fold_y_train = _target[train_index]
            fold_X_test = _input[test_index]
            fold_y_test = _target[test_index]

            metric = model.fit(fold_X_train, fold_y_train, **kwargs)

            metric_model.append_metric(metric)

        return metric_model
