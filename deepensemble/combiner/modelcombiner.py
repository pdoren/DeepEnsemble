from collections import OrderedDict

from ..utils.serializable import Serializable

__all__ = ['ModelCombiner']


class ModelCombiner(Serializable):
    """ Base class for mixing output of models.

    Attributes
    ----------
    _param : dict
        Parameters of combiner method.

    _type_model : str, "regressor" by default
        Type of model: regressor or classifier

    Parameters
    ----------
    param : dict
        Parameters of combiner method.

    type_model : str
        Type of model: regressor or classifier
    """
    # noinspection PyUnusedLocal
    def __init__(self, param=None, type_model="regressor"):
        super(ModelCombiner, self).__init__()
        self._param = {'name': 'Combiner', 'value': None, 'shape': None, 'init': False, 'include': False}
        self._type_model = type_model

    def get_type_model(self):
        """ Get type of model.

        Returns
        -------
        str
            Returns one string with the type of model: "regressor" or "classifier"
        """
        return self._type_model

    def get_param(self, only_values=False):
        """ Getter model combinator parameters.

        Returns
        -------
        theano.shared
            Returns model parameters.
        """
        if not only_values:
            return self._param
        else:
            params = []
            if self._param is not None and self._param['include']:
                params.append(self._param['value'])

            return params

    def output(self, ensemble_model, _input, prob):
        """ Mixing the output or prediction of ensemble's models.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.tensor.matrix
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
        return self.output(ensemble_model, _input, prob=False).eval()

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
