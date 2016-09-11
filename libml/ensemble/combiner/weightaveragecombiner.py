import theano.tensor as T
from .modelcombiner import ModelCombiner
from libml.utils.utils_classifiers import *
from theano import shared

__all__ = ['WeightAverageCombiner']


class WeightAverageCombiner(ModelCombiner):
    def __init__(self, n_models):
        """ Class for compute the average the output models.

        Parameters
        ----------
        n_models : int
            Number of models of ensemble.
        """
        super().__init__()
        self.n_models = n_models

        self.weight = []

        for i in range(n_models):
            self.weight.append(shared(0.0))

    # noinspection PyMethodMayBeStatic
    def output(self, list_models_ensemble, _input):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        list_models_ensemble : numpy.array
            List of models.

        _input : theano.tensor.matrix
            Input sample.

        Returns
        -------
        numpy.array
            Returns the average of the output models.
        """
        output = 0.0
        # TODO: must be optimized performance
        for i, model in enumerate(list_models_ensemble):
            output += model.output(_input) * self.weight[i]
        return output

    def update_parameters(self, ensemble_model, _input, _target):
        """  Update internal parameters.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix
            Input sample.

        _target : theano.tensor.matrix
            Target sample.
        """
        if ensemble_model.type_model is 'classifier':
            _target = translate_target(_target=_target, n_classes=ensemble_model.n_output,
                                       target_labels=ensemble_model.target_labels)

        sumW = 0.0
        for i, model in enumerate(ensemble_model.list_models_ensemble):
            _output = model.output(_input)
            me = T.mean(_output - _target)
            self.weight[i] += me
            sumW += me

        for i, model in enumerate(ensemble_model.list_models_ensemble):
            self.weight[i] = T.constant(1.0) - self.weight[i] / sumW
