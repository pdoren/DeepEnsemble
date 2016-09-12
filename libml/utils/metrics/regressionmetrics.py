import theano.tensor as T
from .basemetrics import *

__all__ = ['RegressionMetrics', 'EnsembleRegressionMetrics', 'score_rms']


class RegressionMetrics(BaseMetrics):
    def __init__(self, model):
        """ Class for generate different metrics for regressor models.

        Parameters
        ----------
        model : Model
            Model.
        """
        super(RegressionMetrics, self).__init__(model=model)


class EnsembleRegressionMetrics(RegressionMetrics, EnsembleMetrics):
    def __init__(self, model):
        """ Class for generate different metrics for ensemble regressor models.

        Parameters
        ----------
        model : Ensemble Model
            Ensemble Model.
        """
        super(EnsembleRegressionMetrics, self).__init__(model=model)


def score_rms(_output, _target):
    """ Gets Root Mean Square like score in a regressor model.

    Parameters
    ----------
    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Returns Root Mean Square.
    """
    e = _output - _target
    return T.mean(T.power(e, 2.0))
