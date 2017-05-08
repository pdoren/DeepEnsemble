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

    def get_max_min_rms(self, decimals=4):
        list_classifiers = []
        for _model in self._model.get_models():
            list_classifiers.append(self._y_pred_per_model[_model.get_name()])

        list_classifiers = np.array(list_classifiers)
        data = []
        for i, target in enumerate(self._y_true_per_model):
            accu = [mean_squared_error(target, pred) for pred in list_classifiers[:, i, :]]
            max_accu = np.around(max(accu), decimals=decimals)
            min_accu = np.around(min(accu), decimals=decimals)
            # noinspection PyTypeChecker
            accu_ensemble = np.around(mean_squared_error(self._y_true[i], self._y_pred[i]), decimals=decimals)
            data += [(max_accu, min_accu, accu_ensemble)]
        return data
