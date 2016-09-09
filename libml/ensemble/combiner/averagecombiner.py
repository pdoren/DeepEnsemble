import theano.tensor as T
from .modelcombiner import ModelCombiner


class AverageCombiner(ModelCombiner):
    def __init__(self):
        """ Class for compute the average the output models.
        """
        super().__init__()

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
        for model in list_models_ensemble:
            output += model.output(_input)
        n = T.constant(len(list_models_ensemble))
        return output / n

    def update_parameters(self, error_models):
        """ Update internal parameters.

        Notes
        -----
        Nothing is done because this class does'nt have parameters.

        Parameters
        ----------
        error_models: theano.tensor.matrix
            Training error models.
        """
        pass
