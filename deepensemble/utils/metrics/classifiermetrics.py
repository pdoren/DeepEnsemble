from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import theano.tensor as T
import numpy as np
from .basemetrics import *
from .diversitymetrics import *
from sklearn.metrics import classification_report
from ...utils.logger import Logger

__all__ = ['ClassifierMetrics', 'EnsembleClassifierMetrics', 'score_accuracy']


class ClassifierMetrics(BaseMetrics):
    """ Class for generate different metrics for classifier models.

    Attributes
    ----------
    y_true : list[numpy.array]
        List of array with target samples.

    y_pred : list[numpy.array]
        List of array with output or prediction of model.

    Parameters
    ----------
    model : Model
        Model.
    """
    def __init__(self, model):
        super(ClassifierMetrics, self).__init__(model=model)
        self.y_pred = []
        self.y_true = []

    def classification_report(self):
        y_true = np.concatenate(tuple(self.y_true))
        y_pred = np.concatenate(tuple(self.y_pred))

        Logger().print(classification_report(y_true, y_pred, target_names=np.char.mod("%s", self.model.target_labels)))

    def append_prediction(self, _target, _output):
        """ Add a sample of prediction and target for generating metrics.

        Parameters
        ----------
        _target : numpy.array
            Target sample.

        _output : numpy.array
            Prediction of the classifier model.

        """
        _output = np.squeeze(_output)
        self.y_pred += [_output]

        _target = np.squeeze(_target)
        self.y_true += [_target]

    def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):
        """ Generate Confusion Matrix plot.

        Parameters
        ----------
        title : str
            Plot title.

        cmap : plt.cm
            Plot Color.

        Notes
        -----
        Show Confusion Matrix plot.

        """
        if len(self.y_pred) > 0 and len(self.y_true) > 0:
            y_true = np.concatenate(tuple(self.y_true))
            y_pred = np.concatenate(tuple(self.y_pred))
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.model.target_labels)
            # normalize
            row_sums = cm.sum(axis=0)
            cm = cm / row_sums[:, np.newaxis]

            f, ax = plt.subplots()
            ax.set_aspect(1)
            res = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            width, height = cm.shape
            for x in range(width):
                for y in range(height):
                    ax.annotate("%*.*f" % (1, 2, cm[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center')
            plt.title(title)
            tick_marks = np.arange(len(self.model.target_labels))
            plt.xticks(tick_marks, self.model.target_labels, rotation=45)
            plt.yticks(tick_marks, self.model.target_labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            res.set_clim(vmin=0, vmax=1)
            plt.colorbar(res)


class EnsembleClassifierMetrics(ClassifierMetrics, EnsembleMetrics):
    """ Class for generate different metrics for ensemble classifier models.

    Parameters
    ----------
    model : Ensemble Model
        Ensemble Model.
    """
    def __init__(self, model):
        super(EnsembleClassifierMetrics, self).__init__(model=model)

    def diversity_report(self):
        metrics = {'Correlation Coefficient': correlation_coefficient,
                   'Kappa statistic': kappa_statistic,
                   'Q statistic': q_statistic,
                   'Double fault': double_fault_measure,
                   'Disagreement': disagreement_measure
                   }

        len_cell = 0
        for model in self.model.list_models_ensemble:
            l = len(model.name)
            if l > len_cell:
                len_cell = l
        cell_format1 = '{0: <%d}' % (len_cell + 3)
        cell_format2 = '{0: >%d}   ' % len_cell
        header = cell_format1.format(' ')
        for model in self.model.list_models_ensemble:
            header += cell_format1.format(model.name)
        line = "-" * len(header)

        for name_metric in sorted(metrics.keys()):
            Logger().print("%s:" % name_metric)
            Logger().print(line)
            Logger().print(header)
            Logger().print(line)
            metric = metrics[name_metric]
            for model1 in self.model.list_models_ensemble:
                Logger().print(cell_format1.format(model1.name), end="")
                list_c1 = self.y_pred_per_model[model1.name]
                for model2 in self.model.list_models_ensemble:
                    list_c2 = self.y_pred_per_model[model2.name]
                    value = "%*.*f" % (2, 4, (self.mean_metric(metric, self.y_true_per_model, list_c1, list_c2)))
                    Logger().print(cell_format2.format(value), end="")
                Logger().print("")  # new line
            Logger().print("")  # new line

    @staticmethod
    def mean_metric(metric, list_target, list_c1, list_c2):
        sum_m = 0.0
        for i, target in enumerate(list_target):
            sum_m += metric(target, list_c1[i], list_c2[i])
        return sum_m / len(list_target)


def score_accuracy(_output, _target):
    """ Accuracy score in a classifier models.

    Parameters
    ----------
    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Returns accuracy in a classifier models.
    """
    return T.mean(T.eq(_output, _target))
