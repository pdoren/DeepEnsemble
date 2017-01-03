from .basemetrics import BaseMetrics, EnsembleMetrics
from sklearn.metrics import mean_squared_error
import numpy as np

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

    def get_max_min_rms(self):
        list_classifiers = []
        for _model in self._model.get_models():
            list_classifiers.append(self._y_pred_per_model[_model.get_name()])

        list_classifiers = np.array(list_classifiers)
        max_accu = 0.0
        min_accu = np.inf
        for i, target in enumerate(self._y_true_per_model):
            accu = mean_squared_error(target, list_classifiers[:, i, :])
            max_accu = max(accu, max_accu)
            min_accu = min(accu, min_accu)
        return min_accu, max_accu
