from collections import OrderedDict

import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .basemetrics import *
from .diversitymetrics import *
from ..utils import Logger

__all__ = ['ClassifierMetrics', 'EnsembleClassifierMetrics']


class ClassifierMetrics(BaseMetrics):
    """ Class for generate different metrics for classifier models.

    Parameters
    ----------
    model : Model
        Model.
    """

    def __init__(self, model):
        super(ClassifierMetrics, self).__init__(model=model)

    def classification_report(self, name_report="Classification Report"):
        """ Generate a classification report (wrapper classification_report scikit)

        Returns
        -------
        None
        """
        precision = None
        recall = None
        f1_score = None
        support = None

        for y_true, y_pred in zip(self._y_true, self._y_pred):
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                          labels=self._model.get_target_labels())

            precision = p[np.newaxis, :] if precision is None else np.concatenate((precision, p[np.newaxis, :]))
            recall = r[np.newaxis, :] if recall is None else np.concatenate((recall, r[np.newaxis, :]))
            f1_score = f1[np.newaxis, :] if f1_score is None else np.concatenate((f1_score, f1[np.newaxis, :]))
            support = s[np.newaxis, :] if support is None else np.concatenate((support, s[np.newaxis, :]))

        metrics = OrderedDict()

        metrics['Precision'] = (np.mean(precision * 100, axis=0), np.std(precision * 100, axis=0))
        metrics['Recall'] = (np.mean(recall * 100, axis=0), np.std(recall * 100, axis=0))
        metrics['f1 Score'] = (np.mean(f1_score * 100, axis=0), np.std(f1_score * 100, axis=0))
        metrics['Support'] = np.mean(support, axis=0)

        len_cell = 0
        for target_label in self._model.get_target_labels():
            l = len(target_label)
            if l > len_cell:
                len_cell = l

        len_cell = max(len_cell, 16)
        cell_format1 = '{0: <%d}' % (len_cell + 2)
        header = cell_format1.format(' ')

        for metric in metrics:
            header += cell_format1.format(metric)
        line = "-" * max(len(header), (len_cell + 3) * len(metrics))

        Logger().print("%s:" % name_report)
        Logger().print(line)
        Logger().print(header)
        Logger().print(line)
        for i, target_label in enumerate(self._model.get_target_labels()):
            Logger().print(cell_format1.format(target_label), end="")
            for key in metrics:
                if key == 'Support':
                    value = "%d" % metrics[key][i]
                else:
                    value = "%.2f +-%.2f" % (metrics[key][0][i], metrics[key][1][i])
                Logger().print(cell_format1.format(value), end="")
            Logger().print("")  # new line
        Logger().print(line)
        Logger().print("")  # new line

    def plot_confusion_matrix(self, **kwargs):
        """ Generate Confusion Matrix plot.

        .. note:: Show Confusion Matrix plot.

        Parameters
        ----------
        kwargs
        """
        if len(self._y_pred) > 0 and len(self._y_true) > 0:
            y_true = np.concatenate(tuple(self._y_true))
            y_pred = np.concatenate(tuple(self._y_pred))
            return self.plot_confusion_matrix_prediction(y_true=y_true, y_pred=y_pred, **kwargs)
        else:
            f, _ = plt.subplots()
            return f

    @staticmethod
    def plot_matrix(ax, cm, labels, title, cmap=plt.cm.Blues):
        """  Plot Matrix.

        Parameters
        ----------
        ax
            Handler plot.

        cm : numpy.array
            Matrix for plotting, the matrix must be square.

        labels : list[str]
            List of labels for each row and cols in Matrix.

        title : str
            Title of plot.

        cmap
            Colors of plot.

        Raises
        ------
        ValueError
        If the matrix is not square.
        """

        if cm.shape[0] != cm.shape[1]:
            raise ValueError('The matrix must be square.')

        ax.set_aspect(1)
        res = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        width, height = cm.shape
        for x in range(width):
            for y in range(height):
                ax.annotate("%*.*f" % (1, 2, cm[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')
        plt.title(title)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        res.set_clim(vmin=0, vmax=1)
        plt.grid()
        plt.colorbar(res)

    def plot_confusion_matrix_prediction(self, y_true, y_pred, title='Confusion matrix', cmap=plt.cm.Blues):
        """ Generate Confusion Matrix plot.

        .. note:: Show Confusion Matrix plot.

        Parameters
        ----------
        y_true : numpy.array
            Target sample.

        y_pred : numpy.array
            Prediction.

        title : str
            Plot title.

        cmap : plt.cm
            Plot Color.
        """
        f, ax = plt.subplots()

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self._model.get_target_labels())

        # normalize
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        self.plot_matrix(ax, cm, self._model.get_target_labels(), title, cmap)

        return f


class EnsembleClassifierMetrics(ClassifierMetrics, EnsembleMetrics):
    """ Class for generate different metrics for ensemble classifier models.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble Model.
    """

    def __init__(self, model):
        super(EnsembleClassifierMetrics, self).__init__(model=model)

    def diversity_report(self):
        """ Generate diversity report of ensemble model.

        Returns
        -------
        None
        """
        metrics = {'Correlation Coefficient': correlation_coefficient,
                   'Kappa statistic': kappa_statistic,
                   'Q statistic': q_statistic,
                   'Double fault': double_fault_measure,
                   'Disagreement': disagreement_measure
                   }

        len_cell = 0
        for model in self._model.get_models():
            l = len(model.get_name())
            if l > len_cell:
                len_cell = l
        len_cell = max(len_cell, 16)
        cell_format1 = '{0: <%d}' % (len_cell + 3)
        cell_format2 = '{0: >%d}   ' % len_cell
        header = cell_format1.format(' ')

        for model in self._model.get_models():
            header += cell_format1.format(model.get_name())
        line = "-" * max(len(header), (len_cell + 3) * self._model.get_num_models())

        for name_metric in sorted(metrics.keys()):
            Logger().print("%s:" % name_metric)
            Logger().print(line)
            Logger().print(header)
            Logger().print(line)
            metric = metrics[name_metric]
            for model1 in self._model.get_models():
                Logger().print(cell_format1.format(model1.get_name()), end="")
                list_c1 = self._y_pred_per_model[model1.get_name()]
                for model2 in self._model.get_models():
                    list_c2 = self._y_pred_per_model[model2.get_name()]
                    mean, std = self.statistic_metric(metric, self._y_true_per_model, list_c1, list_c2)
                    value = "%+.4f +-%.4f" % (mean, std)
                    Logger().print(cell_format2.format(value), end="")
                Logger().print("")  # new line
            Logger().print("")  # new line

    @staticmethod
    def statistic_metric(metric, list_target, list_c1, list_c2):
        """

        Parameters
        ----------
        metric
            Diversity metric.

        list_target : list or array
            List of target.

        list_c1 : list or array
            Prediction first model.

        list_c2 : list or array
            Prediction second model.

        Returns
        -------
        tuple
            Returns average between two metric models.
        """
        data = []
        for i, target in enumerate(list_target):
            data += [metric(target, list_c1[i], list_c2[i])]
        return np.mean(data), np.std(data)

    def plot_diversity(self, input_test, target_test, prefix='diversity'):
        """ Generate a plot for different diversity metrics.

        Parameters
        ----------
        input_test : numpy.array
            Input sample.

        target_test : numpy.array
            Target sample.

        prefix : str
            This string is add as prefix a each label the figures.

        Returns
        -------
        list
            Returns a list of tuple where each tuple has the figure and a label that identify the figure.
        """
        metrics = {'Correlation Coefficient': correlation_coefficient,
                   'Kappa statistic': kappa_statistic,
                   'Q statistic': q_statistic,
                   'Double fault': double_fault_measure,
                   'Disagreement': disagreement_measure
                   }

        N = len(self._model.get_models())

        predictions = []
        labels = []
        for model in self._model.get_models():
            pred_test = model.predict(input_test)
            predictions.append(pred_test)
            labels.append(model.get_name())

        figures = []

        for k, key in enumerate(metrics):
            f, ax = plt.subplots()

            metric = metrics[key]
            cm = np.zeros(shape=(N, N))
            for i in range(N):
                for j in range(N):
                    cm[i, j] = metric(target_test, predictions[i], predictions[j])

            self.plot_matrix(ax, cm, labels, '%s: %s' % (self._model.get_name(), key))

            figures.append((f, '%s_%s' % (prefix, key)))

        return figures
