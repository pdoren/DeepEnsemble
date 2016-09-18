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
        f, ax = plt.subplots()
        self.plot(ax, self.train_cost, max_epoch)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()
        plt.xlabel('epoch')

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
        self.plot(ax, self.train_score, max_epoch)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()
        plt.xlabel('epoch')

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

    def plot(self, ax, dps, max_epoch):
        """ Generate plot.

        Parameters
        ----------
        ax
            Handle subplot.

        dps : list[DataPlot]
            List of DataPlots.

        max_epoch : int
            Number max epoch training.
        """
        # Get average plots
        dpa = DataPlot(self.model.name)
        dpa.data = np.zeros(dps[0].data.shape)
        for dp in dps:
            dpa.data += dp.data
        dpa.data /= len(dps)
        dpa.plot(ax, max_epoch)


class EnsembleMetrics(BaseMetrics):
    """ Class for generate different metrics for ensemble models.

    Attributes
    ----------
    metrics_models : Dict[BaseMetrics]
        Dict of models' metrics.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble Model.
    """
    def __init__(self, model):
        super(EnsembleMetrics, self).__init__(model=model)
        self.metrics_models = {}

    def append_metric(self, metric):
        """ Adds metric of another metric model.

        Parameters
        ----------
        metric : EnsembleMetrics or BaseMetrics
            Metric of another model.
        """
        if isinstance(metric, EnsembleMetrics):
            self.metrics_models.update(metric.metrics_models)
            self.train_cost += metric.train_cost
            self.test_cost += metric.test_cost
            self.train_score += metric.train_score
        else:
            if metric.model.name is self.metrics_models:
                self.metrics_models[metric.model.name].append_metric(metric)
            else:
                self.metrics_models[metric.model.name] = metric

    def plot_cost_models(self, max_epoch, train_title='Train Cost', log_scale=False):
        """ Generate training cost plot for each models in Ensemble.

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
        plt.hold(True)
        for name in sorted(self.metrics_models):
            self.metrics_models[name].plot(ax, self.metrics_models[name].train_cost, max_epoch)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()
        plt.xlabel('epoch')
        plt.hold(False)

    def plot_score_models(self, max_epoch, train_title='Train score', log_scale=False):
        """ Generate training score plot for each models in Ensemble.

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
        plt.hold(True)
        for name in sorted(self.metrics_models):
            self.metrics_models[name].plot(ax, self.metrics_models[name].train_score, max_epoch)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        plt.grid()
        plt.xlabel('epoch')
        plt.hold(False)
