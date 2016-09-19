from .modelcombiner import ModelCombiner

__all__ = ['AverageCombiner']


class AverageCombiner(ModelCombiner):
    """ Class for compute the average the output models.
    """
    def __init__(self):
        super(AverageCombiner, self).__init__()

    # noinspection PyMethodMayBeStatic
    def output(self, ensemble_model, _input):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Returns the average of the output models.
        """
        output = 0.0
        for model in ensemble_model.list_models_ensemble:
            output += model.output(_input)
        n = len(ensemble_model.list_models_ensemble)
        return output / n
