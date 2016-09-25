import numpy as np
import matplotlib.pylab as plt

__all__ = ['BaseMetrics', 'EnsembleMetrics', 'FactoryMetrics']


class FactoryMetrics:
    """ Factory Metrics
    """

    def __init__(self):
        pass

    @staticmethod
    def get_metric(_model):
        """

        Parameters
        ----------
        _model : Model

        Returns
        -------

        """
        return _model.get_new_metric()


class DataPlot:
    """ Class for save data plot.

    Attributes
    ----------

    __x : list[float]
        Axis x.

    __y : list[float]
        Axis y.

    name : str
        Plot's name.

    Parameters
    ----------
    name : str, "Model" by default
        Plot's name.
    """

    def __init__(self, name="Model"):
        self.__x = []
        self.__y = []
        self.__name = name

    def reset(self):
        """ Reset data.
        """
        self.__x = []
        self.__y = []

    def get_name(self):
        """ Get name plot.

        Returns
        -------
        str
            This string is used for plot legend.
        """
        return self.__name

    def set_data(self, x, y):
        """

        Parameters
        ----------
        x : list[float]
            List of points x axis.

        y : list[float]
            List of points y axis.
        """
        self.__x = x
        self.__y = y

    def get_data(self):
        """ Get list of points.

        Returns
        -------
        tuple
            Returns one tuple with 2 list: x and y axis (x, y).
        """
        return self.__x, self.__y

    def len_data(self):
        """ gets count of points.

        Returns
        -------
        int
            Returns counts of points.
        """
        return len(self.__x)

    def add_point(self, x, y):
        """ Adds data.

        Parameters
        ----------
        x : float
            New data.

        y : float
            New data.
        """
        self.__x.append(x)
        self.__y.append(y)

    def plot(self, ax):
        """ Plot data.

        Parameters
        ----------
        ax
            Handle subplot.
        """
        if len(self.__y) > 0:
            ax.plot(self.__x, self.__y, label=self.__name)


class BaseMetrics:
    """ Base class generate metrics.

    Attributes
    ----------
    model : Model or EnsembleModel
        Handle of model.

    _train_cost : list[DataPlot]
        Plot of training cost.

    _train_score : list[DataPlot]
        Plot of prediction score.

    _test_cost : list[DataPlot]
        Plot of testing cost.

    Parameters
    ----------
    model : Model
        Model.
    """

    def __init__(self, model):
        self._model = model
        self._train_cost = []
        self._train_score = []
        self._test_cost = []
        self._test_score = []

    def get_train_cost(self):
        """ Getter train cost.

        Returns
        -------
        list
            Returns train cost list.
        """
        return self._train_cost

    def get_train_score(self):
        """ Getter train cost.

        Returns
        -------
        list
            Returns train score list.
        """
        return self._train_score

    def get_test_cost(self):
        """ Getter test cost.

        Returns
        -------
        list
            Returns test cost list.
        """
        return self._test_cost

    def get_test_score(self):
        """ Getter test cost.

        Returns
        -------
        list
            Returns test score list.
        """
        return self._test_score

    def reset(self):
        """ Reset metrics.
        """
        self._train_cost = []
        self._train_score = []
        self._test_cost = []
        self._test_score = []

    def append_data(self, data, epoch, type_set_data):
        """ Append data in each list.

        Parameters
        ----------
        data : list
            List of data.

        epoch : int
            Current epoch of training when was called this method.

        type_set_data : str
            This string means what kind of data is passed: train or test.

        Returns
        -------
        int
            Returns index of last item saved from data list.
        """
        if type_set_data == "train":
            self.add_point(self._train_cost, epoch, data[0], "train")
        elif type_set_data == "test":
            self.add_point(self._test_cost, epoch, data[0], "test")
        else:
            raise ValueError("The type set data must be 'train' or 'test'.")

        n = 1
        for _ in self._model.get_score_function_list():
            if type_set_data == "train":
                self.add_point(self._train_score, epoch, data[n], "train")
            elif type_set_data == "test":
                self.add_point(self._test_score, epoch, data[n], "test")
            else:
                n -= 1
            n += 1
        return n

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
        plt.hold(True)
        self.plot(ax, self._train_cost)
        self.plot(ax, self._test_cost)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        ax.set_xlim([0, max_epoch])
        plt.grid()
        plt.xlabel('epoch')
        plt.hold(False)

    def plot_score(self, max_epoch, train_title='Train score', log_scale=False, vmin=0, vmax=1):
        """ Generate training score plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        train_title : str, "Train score" by default
            Plot title of training score.

        log_scale : bool, False by default
            Flag for show plot in logarithmic scale.

        vmin : float
            Minimum value shown on the y-axis.

        vmax : float
            Maximum value shown on the y-axis.
        """
        f, ax = plt.subplots()
        plt.hold(True)
        self.plot(ax, self._train_score)
        self.plot(ax, self._test_score)
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        ax.legend()
        ax.set_ylim([vmin, vmax])
        ax.set_xlim([0, max_epoch])
        plt.grid()
        plt.xlabel('epoch')
        plt.hold(False)

    def append_metric(self, metric):
        """ Adds metric of another metric model.

        Parameters
        ----------
        metric : BaseMetrics
            Metric of another model.
        """
        self._train_cost += metric._train_cost
        self._train_score += metric._train_score
        self._test_cost += metric._test_cost

    @staticmethod
    def add_point(list_points, x, y, legend):
        """ Add point a list.

        Parameters
        ----------
        list_points : list[DataPlot]
            List of points.

        x : float
            Point axis x.

        y : float
            Point axis y.

        legend : str
            This is the legend of plot.
        """
        if len(list_points) <= 0:
            list_points.append(DataPlot(name=legend))

        list_points[0].add_point(x, y)

    def plot(self, ax, dps):
        """ Generate plot.

        Parameters
        ----------
        ax
            Handle subplot.

        dps : list[DataPlot]
            List of DataPlots.
        """
        # Get average plots
        if len(dps) > 0:
            dpa = DataPlot("%s %s" % (self._model.get_name(), dps[0].get_name()))
            y = np.zeros((dps[0].len_data(),))
            x = None
            for dp in dps:
                x, y1 = dp.get_data()
                y += np.array(y1)
            dpa.set_data(x, y / len(dps))
            dpa.plot(ax)


class EnsembleMetrics(BaseMetrics):
    """ Class for generate different metrics for ensemble models.

    Attributes
    ----------
    _metrics_models : dict[BaseMetrics]
        Dictionary of models metrics.

    _y_true_per_model : list[numpy.array]
        Array for saving target of sample.

    _y_pred_per_model : dict[numpy.array]
        Dictionary for saving prediction of ensemble models.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble Model.
    """

    def __init__(self, model):
        super(EnsembleMetrics, self).__init__(model=model)
        self._metrics_models = {}
        self._y_true_per_model = []
        self._y_pred_per_model = {}

    def append_data(self, data, epoch, type_set_data):
        n = super(EnsembleMetrics, self).append_data(data, epoch, type_set_data=type_set_data)

        if len(data) > n:
            for model_ensemble in self._model.get_models():
                s_model = model_ensemble.get_name()
                if model_ensemble.get_name() not in self._metrics_models:
                    self._metrics_models[s_model] = FactoryMetrics().get_metric(model_ensemble)

                if type_set_data == "train":
                    self._metrics_models[s_model]. \
                        add_point(self._metrics_models[model_ensemble.get_name()].get_train_cost(), epoch, data[n],
                                  "train")
                    self._metrics_models[s_model]. \
                        add_point(self._metrics_models[model_ensemble.get_name()].get_train_score(), epoch, data[n + 1],
                                  "train")

                elif type_set_data == "test":
                    self._metrics_models[s_model]. \
                        add_point(self._metrics_models[model_ensemble.get_name()].get_test_cost(), epoch, data[n],
                                  "test")
                    self._metrics_models[s_model]. \
                        add_point(self._metrics_models[model_ensemble.get_name()].get_test_score(), epoch, data[n + 1],
                                  "test")
                else:
                    n -= 2
                n += 2
        return n

    def append_prediction_per_model(self, _input, _target):
        _target = np.squeeze(_target)

        self._y_true_per_model += [_target]

        for model_ensemble in self._model.get_models():
            output = np.squeeze(model_ensemble.predict(_input))
            if model_ensemble.get_name() not in self._y_pred_per_model:
                self._y_pred_per_model[model_ensemble.get_name()] = []
            self._y_pred_per_model[model_ensemble.get_name()] += [output]

    def append_metric(self, metric):
        """ Adds metric of another metric model.

        Parameters
        ----------
        metric : EnsembleMetrics or BaseMetrics
            Metric of another model.
        """
        if isinstance(metric, EnsembleMetrics):
            self._metrics_models.update(metric._metrics_models)
            self._train_cost += metric._train_cost
            self._test_cost += metric._test_cost

            self._train_score += metric._train_score
            self._test_score += metric._test_score
        else:
            if metric._model.get_name() in self._metrics_models:
                self._metrics_models[metric._model.get_name()].append_metric(metric)
            else:
                self._metrics_models[metric._model.get_name()] = metric

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
        flag_legend = False
        for name in sorted(self._metrics_models):
            if len(self._metrics_models[name].get_train_cost()) > 0:
                flag_legend = True
                self._metrics_models[name].plot(ax, self._metrics_models[name].get_train_cost())
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        if flag_legend:
            ax.legend()
        plt.grid()
        ax.set_xlim([0, max_epoch])
        plt.xlabel('epoch')
        plt.hold(False)

    def plot_score_models(self, max_epoch, train_title='Train score', log_scale=False, vmin=0, vmax=1):
        """ Generate training score plot for each models in Ensemble.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        train_title : str, "Train score" by default
            Plot title of training score.

        log_scale : bool, False by default
            Flag for show plot in logarithmic scale.

        vmin : float
            Minimum value shown on the y-axis.

        vmax : float
            Maximum value shown on the y-axis.
        """
        f, ax = plt.subplots()
        plt.hold(True)
        flag_legend = False
        for name in sorted(self._metrics_models):
            if len(self._metrics_models[name].get_train_score()) > 0:
                flag_legend = True
                self._metrics_models[name].plot(ax, self._metrics_models[name].get_train_score())
        ax.set_title(train_title)
        if log_scale:
            ax.set_xscale('log')
        if flag_legend:
            ax.legend()
        ax.set_ylim([vmin, vmax])
        ax.set_xlim([0, max_epoch])
        plt.grid()
        plt.xlabel('epoch')
        plt.hold(False)
