from collections import OrderedDict
__all__ = ['ModelCombiner']


class ModelCombiner:
    """ Base class for mixing output of models.

    Attributes
    ----------
    params : theano.shared
        Parameters of combiner method.
    """
    def __init__(self):
        self.params = None

    def output(self, ensemble_model, _input):
        """ Mixing the output or prediction of ensemble's models.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

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

        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression.
        """
        return None
