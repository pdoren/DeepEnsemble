import theano.tensor as T
from .modelcombiner import ModelCombiner


class AverageCombiner(ModelCombiner):

    def __init__(self):
        """ Class for compute the average the output models.
        """
        pass

    # noinspection PyMethodMayBeStatic
    def output(self, list_models_ensemble, _input):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        list_models_ensemble: list
            List of models.

        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        numpy.array
        Returns the average of the output models.

        """
        output = 0.0
        for pair in list_models_ensemble:
            output += pair.model.output(_input)
        n = T.constant(len(list_models_ensemble))
        return output / n