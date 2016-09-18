__all__ = ['ModelCombiner']


class ModelCombiner:
    """ Base class for mixing output of models.
    """
    def __init__(self):
        pass

    def output(self, list_models_ensemble, _input):
        """ Mixing the output or prediction of ensemble's models.

        Parameters
        ----------
        list_models_ensemble : list[Model]
            List of models.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Returns the mixing prediction of ensemble's models.
        """
        raise NotImplementedError

    def update_parameters(self, ensemble_model, _input, _target):
        """ Update internal parameters.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix
            Input sample.

        _target : theano.tensor.matrix
            Target sample.
        """
        pass
