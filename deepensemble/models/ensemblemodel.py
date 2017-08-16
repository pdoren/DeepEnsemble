import numpy as np
from collections import OrderedDict

from .model import Model
from .wrapper import Wrapper
from ..metrics import FactoryMetrics, EnsembleClassifierMetrics, EnsembleRegressionMetrics
from ..utils import Logger, score_accuracy, score_rms

from ..utils.utils_translation import TextTranslation

__all__ = ['EnsembleModel']


class EnsembleModel(Model):
    """ Base class Ensemble Model.

    Attributes
    ----------
    __combiner : AverageCombiner
        The class combiner allows to mix the models outputs.

    __list_models_ensemble : list[Model]
        List of the ensemble's models.

    __list_cost_ensemble : list[]
        List of cost in Ensemble.

    _type_training : str
        This parameter means what type of training perform.

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
        self._type_training = None
        self._pre_training = None
        self._params.append(None)  # the first element is reserved for combiner parameters

    def set_pre_training(self, proc_pre_training, params):
        self._pre_training = (proc_pre_training, params)

    def get_model_input(self):
        """ Gets model input.

        Returns
        -------
        theano.tensor
            Returns model input.
        """
        return self._model_input

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

    def get_num_models(self):
        """ Get number of the Ensemble's models

        Returns
        -------
        int
            Returns current number of models in the Ensemble.
        """
        return len(self.__list_models_ensemble)

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
            self._params[0] = combiner.get_param()
        else:
            raise ValueError(TextTranslation().get_str('Error_3') + " %s." % self.__type_model)

    def set_type_training(self, type_training):
        """ Setter type of training

        Parameters
        ----------
        type_training : str
            This parameter means what type of training perform.

        Returns
        -------
        None
        """
        self._type_training = type_training

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
                self.append_score(score_accuracy, TextTranslation().get_str('Accuracy'))
            else:
                self.append_score(score_rms, TextTranslation().get_str('RMS'))

        elif self.__list_models_ensemble[0] == new_model:
            self.__list_models_ensemble.append(new_model)
        else:
            str_error = ''
            if self.__list_models_ensemble[0].model.get_input_shape() != new_model.get_input_shape():
                str_error += TextTranslation().get_str('Error_4_1')
            if self.__list_models_ensemble[0].model.get_output_shape() != new_model.get_output_shape():
                str_error += TextTranslation().get_str('Error_4_2')
            if self.__list_models_ensemble[0].model.type_model != new_model.get_type_model():
                str_error += TextTranslation().get_str('Error_4_3')
            if self.__list_models_ensemble[0].model.get_target_labels() is not new_model.get_target_labels():
                str_error += TextTranslation().get_str('Error_4_4')

            raise ValueError(TextTranslation().get_str('Error_4_5') + str_error[0:-2] + '.')

        self._params += new_model.get_params()

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
        return self.__combiner.output(self, _input, prob)

    def predict(self, _input):
        """ Compute the diversity of model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Return the diversity of model.
        """
        return self.__combiner.predict(self, _input)

    # noinspection PyProtectedMember
    def update_io(self):
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
            raise AssertionError(TextTranslation().get_str('Error_5') + " %s." % self._name)

        self.update_io()

        # append const ensemble for each models
        if len(self.__list_cost_ensemble) > 0:
            for fun_cost, name, params_cost in self.__list_cost_ensemble:
                for i, model in enumerate(self.get_models()):
                    model.append_cost(fun_cost=fun_cost, name=name, ensemble=self, **params_cost)

        cost = []
        cost_ensemble = self.get_cost()
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

        if cost_ensemble == 0:
            cost = sum(cost)
        else:
            cost = cost_ensemble

        update_combiner = self.__combiner.update_parameters(self, self._model_input, self._model_target)
        updates = OrderedDict()
        if update_combiner is not None:
            updates = update_combiner

        # # update for each models
        # for model in self.get_models():
        #    cost_model = cost
        #    error_model = model.get_error()
        #    update_model = model.get_update_function(cost_model, error_model)
        #    updates.update(update_model)

        update_model = self.get_update_function(cost, self.get_error())
        updates.update(update_model)

        # noinspection PyUnresolvedReferences
        ind_er = np.nonzero(extra_results)[0]
        extra_results = list(np.array(extra_results)[ind_er])
        labels_extra_results = list(np.array(labels_extra_results)[ind_er])
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

    def exists_wrapper_models(self):
        """ Determine whether exist a wrapper model in Ensemble.

        Returns
        -------
        bool
            Returns True if exists wrapper models in Ensemble, False otherwise.
        """
        there_are_wrapper_models = False
        for _model in self.__list_models_ensemble:
            if isinstance(_model, Wrapper):
                there_are_wrapper_models = True
                break

        return there_are_wrapper_models

    def is_need_compile_separately(self):
        """ Determine if it is necessary to compile models separately.

        Returns
        bool
            Returns True if it is necessary to compile models separately, False otherwise.
        """
        return self.exists_wrapper_models() or self._type_training is not None

    def compile(self, fast=True, **kwargs):
        """ Prepare training (compile function of Theano).

        Parameters
        ----------
        fast : bool
           Compile model only necessary.

        kwargs
        """
        if not self.is_need_compile_separately():
            for _model in self.get_models():
                _model.reset_compile()

            super(EnsembleModel, self).compile(fast=fast, **kwargs)
        else:
            Logger().start_measure_time(TextTranslation().get_str('Start_Compile') + " %s" % self._name)
            Logger().log_disable()

            self.update_io()
            for _model in self.get_models():
                _model.compile(fast=fast, **kwargs)

            Logger().log_enable()
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
        if self._pre_training is not None:
            self._pre_training[0](self, _input, _target, **self._pre_training[1])

        if self.is_need_compile_separately():
            return self.fit_separate_models(_input=_input, _target=_target, **kwargs)
        else:
            return super(EnsembleModel, self).fit(_input=_input, _target=_target, **kwargs)

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
        # Choose training method
        if self._type_training == 'boosting':
            return self.__fit_boosting(_input, _target, **kwargs)
        elif self._type_training == 'bagging':
            return self.__fit_bagging(_input, _target, **kwargs)
        else:
            return self.__fit_simple(_input, _target, **kwargs)

    def __fit_simple(self, _input, _target, **kwargs):
        """ Train each models one at time.

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
        metrics = FactoryMetrics().get_metric(self)
        Logger().log(TextTranslation().get_str('Simple_training'), flush=True)
        for model in self.__list_models_ensemble:
            Logger().log(TextTranslation().get_str('Training_model') + ' %s ... ' % model.get_name(), end='', flush=True)
            Logger().log_disable()
            metric = model.fit(_input, _target, **kwargs)
            Logger().log_enable()
            Logger().log("| %s: %.4f / %.4f" % (TextTranslation().get_str('score'), model.get_train_score(), model.get_test_score()))

            metrics.append_metric(metric)

        return metrics

    def __fit_bagging(self, _input, _target, ratio_bootstrap=0.9, **kwargs):
        """ Train the Ensemble with Bagging algorithm.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        seeds : list[]

        ratio_bootstrap : int

        kwargs

        Returns
        -------
        MetricsBase
            Returns metrics got in training.
        """
        metrics = FactoryMetrics().get_metric(self)
        Logger().log('Bagging train ', flush=True)

        N = len(_input)

        n_bootstrap = int(ratio_bootstrap * N)

        for i, model in enumerate(self.__list_models_ensemble):
            Logger().log(TextTranslation().get_str('Training_model') + ' %s ... ' % model.get_name(), end='', flush=True)
            Logger().log_disable()

            # Generate bootstrap
            rns = np.random.RandomState()
            index_bootstrap = rns.randint(0, N, n_bootstrap)

            # Training model
            metric = model.fit(_input[index_bootstrap], _target[index_bootstrap], **kwargs)

            Logger().log_enable()
            Logger().log("| %s: %.4f / %.4f" % (TextTranslation().get_str('score'), model.get_train_score(), model.get_test_score()))

            metrics.append_metric(metric)

        return metrics

    def __fit_boosting(self, _input, _target, **kwargs):
        """ Train the Ensemble with Boosting algorithm.

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
        # TODO: need to be completed
        metrics = FactoryMetrics().get_metric(self)
        Logger().log(TextTranslation().get_str('Boosting_training'), flush=True)
        for model in self.__list_models_ensemble:
            Logger().log(TextTranslation().get_str('Training_model') + ' %s ... ' % model.get_name(), end='', flush=True)
            Logger().log_disable()
            metric = model.fit(_input, _target, **kwargs)
            Logger().log_enable()
            Logger().log("| %s: %.4f / %.4f" % (TextTranslation().get_str('score'), model.get_train_score(), model.get_test_score()))

            metrics.append_metric(metric)

        return metrics
