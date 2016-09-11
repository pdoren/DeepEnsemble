import time
from .combiner.averagecombiner import AverageCombiner
from .combiner.modelcombiner import ModelCombiner
from libml.utils.metrics.classifiermetrics import EnsembleClassifierMetrics
from libml.utils.metrics.regressionmetrics import EnsembleRegressionMetrics
from libml.models.model import Model

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
        for model in self.list_models_ensemble:
            model.reset()

    def output(self, _input):
        """ Output of ensemble model.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
            Returns of combiner the outputs of the different the ensemble's models.
        """
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

    def fit(self, _input, _target, max_epoch, validation_jump, verbose=False, **kwargs):
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

        if verbose:
            toc = time.time()
            print("Elapsed time [s]: %f" % (toc - tic))
        self.combiner.update_parameters(self, _input=_input, _target=_target)
        return metrics

    def add_cost_ensemble(self, fun_cost, **kwargs):
        """ Adds cost function for each models in Ensemble.

        Parameters
        ----------
        fun_cost : str
            Name of cost function.

        kwargs
            Other parameters.
        """
        for i, model in enumerate(self.list_models_ensemble):
            model.append_cost(fun_cost=fun_cost, index_current_model=i, ensemble=self, **kwargs)
