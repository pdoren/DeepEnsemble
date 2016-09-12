from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import theano.tensor as T
from theano import config
import numpy as np
from .basemetrics import *

__all__ = ['ClassifierMetrics', 'EnsembleClassifierMetrics', 'score_accuracy']


class ClassifierMetrics(BaseMetrics):
    def __init__(self, model):
        """ Class for generate different metrics for classifier models.

        Parameters
        ----------
        model : Model
            Model.
        """
        super(ClassifierMetrics, self).__init__(model=model)
        self.cm = []

    def append_prediction(self, _target, _output):
        """ Add a sample of prediction and target for generating metrics.

        Parameters
        ----------
        _target : numpy.array
            Target sample.

        _output : numpy.array
            Prediction of the classifier model.

        """
        cm = confusion_matrix(y_true=_target, y_pred=_output, labels=self.model.target_labels)

        self.cm.append(cm)

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
        if len(self.cm) == 0:
            cm = self.cm
        else:
            cm = np.average(self.cm, axis=0)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.model.target_labels))
        plt.xticks(tick_marks, self.model.target_labels, rotation=45)
        plt.yticks(tick_marks, self.model.target_labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


class EnsembleClassifierMetrics(ClassifierMetrics, EnsembleMetrics):
    def __init__(self, model):
        """ Class for generate different metrics for ensemble classifier models.

        Parameters
        ----------
        model : Ensemble Model
            Ensemble Model.
        """
        super(EnsembleClassifierMetrics, self).__init__(model=model)


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
    return T.mean(T.eq(_output, _target), dtype=config.floatX)
