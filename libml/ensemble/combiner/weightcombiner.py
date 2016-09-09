import theano.tensor as T
import numpy as np
from .modelcombiner import ModelCombiner
from theano import shared, config


class WeightCombiner(ModelCombiner):
    def __init__(self, n_models, ratio_update):
        """ Class for compute the average the output models.

        Parameters
        ----------
        n_models: int
            Number of models of ensemble.

        ratio_update: float or double
            Ratio for update weights.

        """
        super().__init__()
        self.weight = shared(np.zeros(shape=(n_models,), dtype=config.floatX), name='weight', borrow=True)
        self.lamb = T.constant(ratio_update, dtype=config.floatX)

    # noinspection PyMethodMayBeStatic
    def output(self, list_models_ensemble, _input):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        list_models_ensemble: numpy.array
            List of models.

        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        numpy.array
        Returns the average of the output models.

        """
        output = 0.0
        sumW = 0.0
        # TODO: must be optimized performance
        for i, model in enumerate(list_models_ensemble):
            output += model.output(_input) * self.weight[i]
            sumW += self.weight[i]
        return output / sumW

    def update_parameters(self, error_models):
        """ Update internal parameters from training error models.

        The weight is proportional of error.

        .. math:: W_{k+1} = W_{k} + \lambda error_models

        Parameters
        ----------
        error_models: theano.tensor.matrix
            Training error models.
        """
        self.weight = self.weight + self.lamb * T.exp(T.power(error_models, 2.0))
