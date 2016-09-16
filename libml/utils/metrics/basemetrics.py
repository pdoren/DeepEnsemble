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

    def add_point(self, new_data):
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

    train_cost : list[DataPlot]
        Plot of training cost.

    train_score : list[DataPlot]
        Plot of prediction score.

    test_cost : list[DataPlot]
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

    def add_point_train_cost(self, point):
        """ Add cost of training.

        Parameters
        ----------
        point : float
            Training cost.
        """
        self.add_point(self.train_cost, point)

    def add_point_train_score(self, point):
        """ Add score of training.

        Parameters
        ----------
        point : float
            Training score.
        """
        self.add_point(self.train_score, point)

    def add_point_test_cost(self, point):
        """ Add cost of testing.

        Parameters
        ----------
        point : float
            Training cost.
        """
        self.add_point(self.test_cost, point)

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

    def append_metric(self, metric):
        """ Adds metric of another metric model.

        Parameters
        ----------
        metric : BaseMetrics
            Metric of another model.
        """
        self.train_cost += metric.train_cost
        self.train_score += metric.train_score
        self.test_cost += metric.test_cost

    def add_point(self, list_points, point):
        """ Add point a list.

        Parameters
        ----------
        list_points : list[DataPlot]
            List of points.

        point : float
            Point.
        """
        if list_points.__len__() <= 0:
            list_points.append(DataPlot(name="%s" % self.model.name))
        else:
            list_points[0].add_point(point)

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

        # Get average plots
        dpa = DataPlot(dps[0].name)
        dpa.data = np.zeros(dps[0].data.shape)
        for dp in dps:
            dpa.data += dp.data
        dpa.data /= len(dps)
        dpa.plot(ax, max_epoch)

        ax.set_title(title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()
        plt.xlabel('epoch')


class EnsembleMetrics(BaseMetrics):
    """ Class for generate different metrics for ensemble models.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble Model.
    """
    def __init__(self, model):
        super(EnsembleMetrics, self).__init__(model=model)

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

        for dp in dps:
            dp.plot(ax, max_epoch)

        ax.set_title(title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()
        plt.xlabel('epoch')
