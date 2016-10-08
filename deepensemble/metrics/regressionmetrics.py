import theano.tensor as T
from .basemetrics import BaseMetrics, EnsembleMetrics

__all__ = ['RegressionMetrics', 'EnsembleRegressionMetrics']


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
