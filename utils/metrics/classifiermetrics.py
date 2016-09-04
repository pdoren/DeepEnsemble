from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import numpy as np


class ClassifierMetrics:
    def __init__(self, target_names):
        """ Class for generate different metrics for classifier models.

        Parameters
        ----------
        target_names: numpy.array or list
            Array with target labels.

        """
        self.metrics = []
        self.cm = []
        self.train_cost = []
        self.test_cost = []
        self.W = 3
        self.target_names = target_names

    def append_pred(self, y_true, y_pred):
        """ Add a sample of prediction and target for generating metrics.

        Parameters
        ----------
        y_true: numpy.array
            Target sample.

        y_pred: numpy.array
            Prediction of the classifier model.

        """
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.target_names)

        self.cm.append(cm)

        TP = cm[0, 0]
        FP = np.sum(cm[1:, 0])
        FN = np.sum(cm[0, 1:])
        TN = np.sum(cm[1:, 1:])

        metrics = np.zeros((1, 4))
        metrics[0, 0] = TP / (TP + FP)
        metrics[0, 1] = TP / (TP + FN)
        metrics[0, 2] = FP / (FP + TN)
        metrics[0, 3] = TP ** 2 / ((TP + self.W * FP) * (TP + FN))

        if len(self.metrics) == 0:
            self.metrics = metrics
        else:
            self.metrics = np.append(self.metrics, metrics, axis=0)

    def append_cost(self, train_cost, test_cost):
        """ Add cost of training.

        Parameters
        ----------
        train_cost: numpy.array
            Training cost.

        test_cost: numpy.array
            Test cost.

        """
        if len(self.train_cost) == 0 and len(self.test_cost) == 0:
            self.train_cost = train_cost
            self.test_cost = test_cost
        else:
            self.train_cost = np.vstack([self.train_cost, train_cost])
            self.test_cost = np.vstack([self.test_cost, test_cost])

    def reset(self):
        """ Reset metrics.
        """
        self.metrics = []
        self.cm = []
        self.train_cost = []
        self.test_cost = []

    def print(self):
        """ Print data metrics.

        Note
        ----
        Show in console information about metrics.

        """
        if len(self.cm) == 0:
            print(self.cm)
        else:
            print(np.average(self.cm, axis=0))

        print("Results:")
        print("Precision:\t %f +- %f" % (np.mean(self.metrics[:, 0]), np.std(self.metrics[:, 0])))
        print("Recall:\t\t %f +- %f" % (np.mean(self.metrics[:, 1]), np.std(self.metrics[:, 1])))
        print("FPR:\t\t %f +- %f" % (np.mean(self.metrics[:, 2]), np.std(self.metrics[:, 2])))
        print("FoM:\t\t %f +- %f" % (np.mean(self.metrics[:, 3]), np.std(self.metrics[:, 3])))

    def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):
        """ Generate Confusion Matrix plot.

        Parameters
        ----------
        title: str
            Plot title.

        cmap: plt.cm
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
        tick_marks = np.arange(len(self.target_names))
        plt.xticks(tick_marks, self.target_names, rotation=45)
        plt.yticks(tick_marks, self.target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_cost(self, max_epoch, log_scale=False, train_title='Train Cost', test_title='Test Cost'):
        """ Generate training cost plot.

        Parameters
        ----------
        max_epoch: int
            Number of epoch of training.

        log_scale: bool
            Flag for show plot in logarithmic scale.

        train_title: str
            Plot title of training cost.

        test_title: str
            Plot title of test cost.

        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(2, 1, 1)
        train_cost = np.transpose(self.train_cost)
        ax.plot(np.linspace(0.0, max_epoch, num=len(train_cost)), train_cost)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        plt.grid()
        ax = fig.add_subplot(2, 1, 2)
        test_cost = np.transpose(self.test_cost)
        ax.plot(np.linspace(0.0, max_epoch, num=len(test_cost)), test_cost)
        ax.set_title(test_title)
        if log_scale:
            ax.set_xscale('log')
        plt.grid()


#
# TEST
#
def test():
    metrics = ClassifierMetrics([0, 1, 2])
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    train_cost = [5, 4, 3, 1, 0.3, 0.2, 0.1]
    test_cost = [15, 20, 5, 1, 0.3, 0.2, 0.1]
    metrics.append_pred(y_true, y_pred)
    metrics.append_cost(train_cost, test_cost)
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 1, 2, 1, 0, 1]
    train_cost = [5, 4, 3, 1, 0.3, 0.2, 0.1]
    test_cost = [5, 2, 5, 7, 0.3, 0.2, 0.1]
    metrics.append_pred(y_true, y_pred)
    metrics.append_cost(train_cost, test_cost)
    metrics.print()
    metrics.plot_confusion_matrix()
    metrics.plot_cost(7)

    plt.show()


if __name__ == "__main__":
    test()
