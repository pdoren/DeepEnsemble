import numpy as np
import matplotlib.pylab as plt

__all__ = ['BaseMetrics', 'EnsembleMetrics']


class DataPlot:
    def __init__(self, name="Model"):
        """ Class for save data plot.

        Parameters
        ----------
        name : str, "Model" by default
            Plot name.
        """
        self.data = np.array([])
        self.name = name

    def reset(self):
        """ Reset data.
        """
        self.data = np.array([])

    def append(self, new_data):
        """ Adds data.

        Parameters
        ----------
        new_data : float
            New data.
        """
        if self.data.size == 0:
            self.data = new_data
        else:
            self.data = np.vstack([self.data, new_data])

    def plot(self, ax, n):
        """ Plot data.

        Parameters
        ----------
        ax
            Handle subplot.

        n : float
            Max limit plot.
        """
        if self.data.size > 0:
            ax.plot(np.linspace(0.0, n, num=len(self.data)), self.data, label=self.name)


class BaseMetrics:
    def __init__(self, model):
        """ Base class generate metrics.

        Parameters
        ----------
        model : Model
            Model.
        """
        self.model = model
        self.train_cost = []
        self.train_score = []
        self.test_cost = []

    def reset(self):
        """ Reset metrics.
        """
        self.train_cost = []
        self.train_score = []
        self.test_cost = []

    def append_train_cost(self, train_cost):
        """ Add cost of training.

        Parameters
        ----------
        train_cost : numpy.array
            Training cost.
        """
        if len(self.train_cost) <= 0:
            self.train_cost.append(DataPlot(name="%s" % self.model.name))
        self.train_cost[0].append(train_cost)

    def append_train_score(self, train_score):
        """ Add score of training.

        Parameters
        ----------
        train_score : numpy.array
            Training score.
        """
        if len(self.train_score) <= 0:
            self.train_score.append(DataPlot(name="%s" % self.model.name))
        self.train_score[0].append(train_score)

    def append_test_cost(self, test_cost):
        """ Add cost of testing.

        Parameters
        ----------
        test_cost : numpy.array
            Training cost.
        """
        if len(self.test_cost) <= 0:
            self.test_cost.append(DataPlot(name="%s" % self.model.name))
        self.test_cost[0].append(test_cost)

    def plot_cost(self, max_epoch, train_title='Train Cost', log_scale=False):
        """ Generate training cost plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        train_title : str
            Plot title of training cost.

        log_scale : bool
            Flag for show plot in logarithmic scale.
        """
        f, ax = plt.subplots()
        for dp in self.train_cost:
            dp[0].plot(ax, max_epoch)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()

    def plot_score(self, max_epoch, train_title='Train score', log_scale=False):
        """ Generate training score plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        train_title : str, "Train score" by default
            Plot title of training score.

        log_scale : bool, False by default
            Flag for show plot in logarithmic scale.
        """
        f, ax = plt.subplots()
        for dp in self.train_score:
            dp[0].plot(ax, max_epoch)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()


class EnsembleMetrics(BaseMetrics):
    def __init__(self, model):
        """ Class for generate different metrics for ensemble models.

        Parameters
        ----------
        model : EnsembleModel
            Ensemble Model.
        """
        super(EnsembleMetrics, self).__init__(model=model)

    def append_metric(self, metric):
        """ Adds metric of another metric model.

        Parameters
        ----------
        metric : BaseMetrics
            Metric of another model.
        """
        self.train_cost.append(metric.train_cost)
        self.train_score.append(metric.train_score)
        self.test_cost.append(metric.test_cost)
