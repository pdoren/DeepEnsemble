import theano.tensor as T
from .basemetrics import BaseMetrics, EnsembleMetrics

__all__ = ['RegressionMetrics', 'EnsembleRegressionMetrics', 'score_rms']


class RegressionMetrics(BaseMetrics):
    """ Class for generate different metrics for regressor models.

    Parameters
    ----------
    model : Model
        Model.
    """
    def __init__(self, model):
        super(RegressionMetrics, self).__init__(model=model)


class EnsembleRegressionMetrics(RegressionMetrics, EnsembleMetrics):
    """ Class for generate different metrics for ensemble regressor models.

    Parameters
    ----------
    model : Ensemble Model
        Ensemble Model.
    """
    def __init__(self, model):
        super(EnsembleRegressionMetrics, self).__init__(model=model)


def score_rms(_input, _output, _target, model):
    """ Gets Root Mean Square like score in a regressor model.

    Parameters
    ----------
    _input : theano.tensor.matrix
        Input sample.

    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    model : Model
        Model.

    Returns
    -------
    theano.tensor.matrix
        Returns Root Mean Square.
    """
    e = _output - _target
    return T.mean(T.power(e, 2.0))
