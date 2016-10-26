from collections import OrderedDict

from .model import Model
from .wrapper import Wrapper
from ..metrics import *
from ..utils import Logger, score_accuracy, score_rms

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
        """ Getter model combiner in Ensemble.

        Returns
        -------
        ModelCombiner
            Returns a models combiner.
        """
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

            # Overwrite current score default
            self._score_function_list = {'list': [], 'changed': True, 'result': []}

            if self.is_classifier():
                self.append_score(score_accuracy, 'Accuracy')
            else:
                self.append_score(score_rms, 'Root Mean Square')

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
        _type = 'prob' if prob else 'crisp'

        if _input == self._model_input:

            if self._output[_type]['changed']:
                self._output[_type]['result'] = self.__combiner.output(self, _input, prob)

            return self._output[_type]['result']

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

    def __update_io(self):
        """ Update Input Output shared Theano variables """
        # Define input-output variables in ensemble
        self._define_input()
        self._define_output()

        # This way all models share the same input-output variables
        for model in self.get_models():
            model._copy_input(self)
            model._copy_output(self)
            model._copy_batch_ratio(self)


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
            raise AssertionError("Not exists combiner method for %s." % self._name)

        self.__update_io()

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

        update_combiner = self.__combiner.update_parameters(self, self._model_input, self._model_target)
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

    def is_fit_ensemble_normal(self):
        """ Determine whether it is possible to train normally.

        Returns
        -------
        bool
            Returns True if it is possible to train normally, False otherwise.
        """
        there_are_wrapper_models = True
        for _model in self.__list_models_ensemble:
            if isinstance(_model, Wrapper):
                there_are_wrapper_models = False
                break

        return there_are_wrapper_models

    def compile(self, fast=True, **kwargs):
        """ Prepare training (compile function of Theano).

        Parameters
        ----------
        fast : bool
           Compile model only necessary.

        kwargs
        """
        if self.is_fit_ensemble_normal():
            super(EnsembleModel, self).compile(fast=fast, **kwargs)
        else:  # TODO: changed when improve Wrapper model
            Logger().start_measure_time("Start Compile %s" % self._name)
            self.__update_io()
            Logger().stop_measure_time()
            self._is_compiled = True

    def fit(self, _input, _target, **kwargs):
        """ Training Ensemble.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        kwargs

        Returns
        -------
        MetricsBase
            Returns metrics got in training.
        """
        if self.is_fit_ensemble_normal():
            return super(EnsembleModel, self).fit(_input=_input, _target=_target, **kwargs)
        else:
            return self.fit_separate_models(_input=_input, _target=_target, **kwargs)

    def fit_separate_models(self, _input, _target, **kwargs):
        """ Training ensemble models each separately.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        kwargs

        Returns
        -------
        MetricsBase
            Returns metrics got in training.
        """
        # create a specific metric
        metrics = FactoryMetrics().get_metric(self)

        for model in self.__list_models_ensemble:

            Logger().log('Training model %s ... ' % model.get_name(), end='', flush=True)
            Logger().log_disable()
            metric = model.fit(_input, _target, **kwargs)
            Logger().log_enable()
            Logger().log("| score: %.4f / %.4f" % (model.get_train_score(), model.get_test_score()))

            metrics.append_metric(metric)

        return metrics
