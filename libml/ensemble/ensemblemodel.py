import time
from ..combiner.averagecombiner import AverageCombiner
from ..models.model import Model
from ..utils.metrics.classifiermetrics import *
from ..utils.metrics.regressionmetrics import *
from ..utils.utils_classifiers import *
import numpy as np

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
        super(EnsembleModel, self).__init__(n_input=0, n_output=0, name=name)
        self.combiner = AverageCombiner()
        self.list_models_ensemble = []

    def set_combiner(self, combiner):
        """ Setter combiner.

        Parameters
        ----------
        combiner : ModelCombiner
            Object ModelCombiner for combining model outputs in ensemble.
        """
        self.combiner = combiner

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
        if len(self.list_models_ensemble) == 0:
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
                self._output = self.combiner.output(self.list_models_ensemble, _input)
            return self._output
        else:
            return self.combiner.output(self.list_models_ensemble, _input)

    def compile(self, **kwargs):
        """ Compile ensemble's models.

        Parameters
        ----------
        kwargs
            Compilers parameters of models.
        """
        for model in self.list_models_ensemble:
            model.compile(**kwargs)

    def fit(self, _input, _target, max_epoch, validation_jump, verbose=False, batch_size=32, **kwargs):
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

        verbose : bool, False by default
            Flag for show training information.

        batch_size: int
            Size of batch.

        kwargs
            Other parameters.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        target_train = _target
        input_train = _input
        metrics = []
        if self.type_model is "classifier":
            metric_ensemble = EnsembleClassifierMetrics(self)
            for model in self.list_models_ensemble:
                metrics.append(ClassifierMetrics(model))
            target_train = translate_target(_target=_target, n_classes=self.n_output,
                                            target_labels=self.target_labels)
        else:
            metric_ensemble = EnsembleRegressionMetrics(self)
            for model in self.list_models_ensemble:
                metrics.append(ClassifierMetrics(model))

        tic = 0.0  # Warning PEP8
        if verbose:
            tic = time.time()

        # Present mini-batches in different order
        rand_perm = np.random.permutation(len(target_train))
        input_train = input_train[rand_perm]
        target_train = target_train[rand_perm]

        for epoch in range(0, max_epoch):
            for i, model in enumerate(self.list_models_ensemble):
                # Train minibatches
                train_cost, train_score = model.minibatch_eval(_input=input_train, _target=target_train,
                                                               batch_size=batch_size, train=True)
                metrics[i].add_point_train_cost(train_cost)
                metrics[i].add_point_train_score(train_score)

        for metric in metrics:
            metric_ensemble.append_metric(metric)

        # update parameters of combiner
        self.combiner.update_parameters(self, _input=_input, _target=_target)

        if verbose:
            toc = time.time()
            print("Elapsed time [s]: %f" % (toc - tic))
        return metric_ensemble

    def fit_diff_training_set(self, _input, _target, max_epoch, validation_jump, verbose=False, **kwargs):
        """ Training ensemble where each model is training with different training set.

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

        verbose : bool, False by default
            Flag for show training information.

        kwargs
            Other parameters.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        if self.type_model is "classifier":
            metrics = EnsembleClassifierMetrics(self)
        else:
            metrics = EnsembleRegressionMetrics(self)

        tic_m, tic = 0.0, 0.0  # Warning PEP8
        if verbose:
            tic = time.time()
            tic_m = tic

        for i, model in enumerate(self.list_models_ensemble):
            metrics.append_metric(model.fit(_input=_input, _target=_target, max_epoch=max_epoch,
                                            validation_jump=validation_jump, **kwargs))
            if verbose:
                toc_m = time.time()
                print("model %i Ok: %f[s]" % (i, toc_m - tic_m))
                tic_m = toc_m

        # update parameters of combiner
        self.combiner.update_parameters(self, _input=_input, _target=_target)

        if verbose:
            toc = time.time()
            print("Elapsed time [s]: %f" % (toc - tic))
        return metrics

    def add_cost_ensemble(self, fun_cost, **kwargs):
        """ Adds cost function for each models in Ensemble.

        Parameters
        ----------
        fun_cost : theano.function
            Cost function.

        kwargs
            Other parameters.
        """
        for i, model in enumerate(self.list_models_ensemble):
            model.append_cost(fun_cost=fun_cost, index_current_model=i, ensemble=self, **kwargs)
