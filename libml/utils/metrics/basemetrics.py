import numpy as np
import matplotlib.pylab as plt

__all__ = ['BaseMetrics', 'EnsembleMetrics']


class DataPlot:
    """ Class for save data plot.

    Attributes
    ----------

    data : numpy.array
        Array save plot's points.

    name : str
        Plot's name.

    Parameters
    ----------
    name : str, "Model" by default
        Plot's name.
    """
    def __init__(self, name="Model"):
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
    """ Base class generate metrics.

    Attributes
    ----------
    model : Model
        Handle of model.

    train_cost : DataPlot
        Plot of training cost.

    train_score : DataPlot
        Plot of prediction score.

    test_cost : DataPlot
        Plot of testing cost.

    Parameters
    ----------
    model : Model
        Model.
    """
    def __init__(self, model):
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

    def append_train_cost(self, point):
        """ Add cost of training.

        Parameters
        ----------
        point : float
            Training cost.
        """
        if len(self.train_cost) <= 0:
            self.train_cost.append(DataPlot(name="%s" % self.model.name))
        self.train_cost[0].append(point)

    def append_train_score(self, point):
        """ Add score of training.

        Parameters
        ----------
        point : float
            Training score.
        """
        if len(self.train_score) <= 0:
            self.train_score.append(DataPlot(name="%s" % self.model.name))
        self.train_score[0].append(point)

    def append_test_cost(self, point):
        """ Add cost of testing.

        Parameters
        ----------
        point : float
            Training cost.
        """
        if len(self.test_cost) <= 0:
            self.test_cost.append(DataPlot(name="%s" % self.model.name))
        self.test_cost[0].append(point)

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
        self.plot(self.train_cost, max_epoch, train_title, log_scale)

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
        self.plot(self.train_score, max_epoch, train_title, log_scale)

    @staticmethod
    def plot(dps, max_epoch, title='Plot', log_scale=False):
        """ Generate plot.

        Parameters
        ----------
        dps : list[DataPlot]
            List of DataPlots.

        max_epoch : int
            Number max epoch training.

        title : str
            Title model.

        log_scale : bool
            Flag for show plot in logarithmic scale.
        """
        f, ax = plt.subplots()
        if len(dps) > 1:
            for dp in dps:
                dp[0].plot(ax, max_epoch)
        else:
            dps[0].plot(ax, max_epoch)
        ax.set_title(title)
        ax.xlabel('epoch')
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()


class EnsembleMetrics(BaseMetrics):
    """ Class for generate different metrics for ensemble models.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble Model.
    """
    def __init__(self, model):
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
