class ModelCombiner:
    def __init__(self):
        """ Base class for mixing output of models.
        """
        pass

    def output(self, list_models_ensemble, _input):
        """ Mixing the output or prediction of ensemble's models.

        Parameters
        ----------
        list_models_ensemble: list
            List of models.

        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        numpy.array
            Returns the mixing prediction of ensemble's models.
        """
        raise NotImplementedError

    def update_parameters(self, error_models):
        """ Update internal parameters.

        Parameters
        ----------
        error_models: theano.tensor.matrix
            Training error models.
        """
        raise NotImplementedError
