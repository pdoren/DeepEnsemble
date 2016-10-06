from collections import OrderedDict
from theano import shared

__all__ = ['ModelCombiner']


class ModelCombiner(object):
    """ Base class for mixing output of models.

    Attributes
    ----------
    _params : theano.shared
        Parameters of combiner method.

    _type_model : str, "regressor" by default
        Type of model: regressor or classifier

    Parameters
    ----------
    params : theano.shared
        Parameters of combiner method.

    type_model : str
        Type of model: regressor or classifier
    """
    def __init__(self, params=shared(0), type_model="regressor"):
        self._params = params
        self._type_model = type_model

    def get_type_model(self):
        """ Get type of model.

        Returns
        -------
        str
            Returns one string with the type of model: "regressor" or "classifier"
        """
        return self._type_model

    def get_params(self):
        """ Getter model combinator parameters.

        Returns
        -------
        theano.shared
            Returns parameters.
        """
        return self._params

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

    def predict(self, ensemble_model, _input):
        """ Compute the prediction of model.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble model where gets the output.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Return the prediction of model.
        """
        return ensemble_model.predict(_input)

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
