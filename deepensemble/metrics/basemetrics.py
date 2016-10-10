import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict

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

    __name : str
        Plot's name.

    __type : str
        Type of data.

    Parameters
    ----------
    name : str, "Model" by default
        Plot's name.
    """

    def __init__(self, name='Model', _type='score'):
        self.__x = []
        self.__y = []
        self.__name = name
        self.__type = _type

    def reset(self):
        """ Reset data.
        """
        self.__x = []
        self.__y = []

    def get_type(self):
        """ Get type plot.

        Returns
        -------
        str
            This string is used for legend and title plot.
        """
        return self.__type

    def get_name(self):
        """ Get name plot.

        Returns
        -------
        str
            This string is used for legend plot.
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
            ax.plot(self.__x, self.__y, label='%s %s' % (self.__type, self.__name))


class BaseMetrics:
    """ Base class generate metrics.

    Attributes
    ----------
    model : Model or EnsembleModel
        Handle of model.

    _error : dict[list[DataPlot]]
        Data plot error.

    _cost : dict[list[DataPlot]]
        Data plot cost.

    _costs : dict[dict[list[DataPlot]]]
        Data plot error.

    _scores : dict[dict[list[DataPlot]]]
        Data plot scores.

    Parameters
    ----------
    model : Model
        Model.
    """

    def __init__(self, model):
        self._model = model
        self._error = {'train': [], 'test': []}
        self._cost = {'train': [], 'test': []}
        self._costs = {'train': OrderedDict(), 'test': OrderedDict()}
        self._scores = {'train': OrderedDict(), 'test': OrderedDict()}

    def get_cost(self, type_set_data):
        """ Getter total cost.

        Parameters
        ----------
        type_set_data : str
            This string means what kind of data is passed: train or test.

        Returns
        -------
        list
            Returns cost list.
        """
        return self._cost[type_set_data]

    def get_costs(self, type_set_data):
        """ Getter costs.

        Parameters
        ----------
        type_set_data : str
            This string means what kind of data is passed: train or test.

        Returns
        -------
        dict
            Returns costs dictionary.
        """
        return self._costs[type_set_data]

    def get_scores(self, type_set_data):
        """ Getter scores.

        Parameters
        ----------
        type_set_data : str
            This string means what kind of data is passed: train or test.

        Returns
        -------
        dict
            Returns scores dictionary.
        """
        return self._scores[type_set_data]

    def reset(self):
        """ Reset metrics.
        """
        self._error = {'train': [], 'test': []}
        self._cost = {'train': [], 'test': []}
        self._costs = {'train': OrderedDict(), 'test': OrderedDict()}
        self._scores = {'train': OrderedDict(), 'test': OrderedDict()}

    def append_data(self, data, epoch, type_set_data):
        """ Append metrics data.

        Parameters
        ----------
        data : list or array
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
        if type_set_data != "train" and type_set_data != "test":
            raise ValueError("The type set data must be 'train' or 'test'.")

        labels = self._model.get_result_labels()

        n = 0  # data[0] is the error
        self.add_point(self._error[type_set_data], epoch, data[n], labels[n], self._model.get_name())

        n += 1  # data[1] is the total cost
        self.add_point(self._cost[type_set_data], epoch, data[n], labels[n], self._model.get_name())

        n = self.add_data(labels, self._model.get_name(),
                          self.get_costs(type_set_data), len(self._model.get_costs()), n, data, epoch)

        return self.add_data(labels, self._model.get_name(),
                             self.get_scores(type_set_data), len(self._model.get_scores()), n, data, epoch)

    @staticmethod
    def add_data(labels, model_name, data_dict, n_data, index, data, epoch):
        for _ in range(n_data):
            index += 1
            label = labels[index]
            if label not in data_dict:
                data_dict[label] = []
            BaseMetrics.add_point(data_dict[label], epoch, data[index], label, model_name)
        return index

    def plot_cost(self, max_epoch, train_title='Cost', log_xscale=False, log_yscale=False):
        """ Generate cost plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        train_title : str
            Plot title of training cost.

        log_xscale : bool
            Flag for show plot x-axis in logarithmic scale.

        log_yscale : bool
            Flag for show plot y-axis in logarithmic scale.
        """
        f, ax = plt.subplots()
        plt.hold(True)
        self.plot(ax, self.get_cost('train'), 'Train')
        self.plot(ax, self.get_cost('test'), 'Test')
        ax.set_title(train_title)
        if log_xscale:
            ax.set_xscale('log')
        if log_yscale:
            ax.set_yscale('log')
        ax.legend(loc='best')
        ax.set_xlim([0, max_epoch])
        plt.xlabel('epoch')
        plt.hold(False)
        plt.tight_layout()

        return f

    def plot_costs(self, max_epoch, title='Cost', log_xscale=False, log_yscale=False):
        """ Generate costs plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        title : str
            Plot title of cost.

        log_xscale : bool
            Flag for show plot x-axis in logarithmic scale.

        log_yscale : bool
            Flag for show plot y-axis in logarithmic scale.
        """
        data_train = self.get_costs('train')
        data_test = self.get_costs('test')
        data = set(data_train) & set(data_test)
        f, _ = plt.subplots()
        N = len(data)
        cols = max(N // 2, 1)
        rows = max(N // cols, 1)
        for j, i in enumerate(data_train):
            ax = plt.subplot(rows, cols, j + 1)
            plt.hold(True)
            BaseMetrics.plot(ax, data_train[i], 'Train')
            BaseMetrics.plot(ax, data_test[i], 'Test')
            plt.hold(False)
            ax.set_title('%s: %s' % (title, data_train[i][0].get_type()))
            if log_xscale:
                ax.set_xscale('log')
            if log_yscale:
                ax.set_yscale('log')
            ax.legend(loc='best')
            ax.set_xlim([0, max_epoch])
            plt.xlabel('epoch')
            plt.tight_layout()

        return f

    def plot_scores(self, max_epoch, title='Score', log_xscale=False, log_yscale=False):
        """ Generate training score plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        title : str, "Train score" by default
            Plot title of training score.

        log_xscale : bool
            Flag for show plot x-axis in logarithmic scale.

        log_yscale : bool
            Flag for show plot y-axis in logarithmic scale.
        """
        data_train = self.get_scores('train')
        data_test = self.get_scores('test')
        data = set(data_train) & set(data_test)
        f, _ = plt.subplots()
        N = len(data)
        cols = max(N // 2, 1)
        rows = max(N // cols, 1)
        for j, i in enumerate(data_train):
            ax = plt.subplot(rows, cols, j + 1)
            plt.hold(True)
            BaseMetrics.plot(ax, data_train[i], 'Train')
            BaseMetrics.plot(ax, data_test[i], 'Test')
            plt.hold(False)
            ax.set_title('%s: %s' % (title, data_train[i][0].get_type()))
            if log_xscale:
                ax.set_xscale('log')
            if log_yscale:
                ax.set_yscale('log')
            ax.legend(loc='best')
            # ax.set_ylim([vmin, vmax])
            ax.set_xlim([0, max_epoch])
            plt.xlabel('epoch')
            plt.tight_layout()

        return f

    def append_metric(self, metric):
        """ Adds metric of another metric model.

        Parameters
        ----------
        metric : BaseMetrics
            Metric of another model.
        """
        for type_set_data in ['train', 'test']:
            self._error[type_set_data] += metric._error[type_set_data]
            self._cost[type_set_data] += metric._cost[type_set_data]
            for key in metric.get_costs(type_set_data):
                if key in self._costs[type_set_data]:
                    self._costs[type_set_data][key] += metric.get_costs(type_set_data)[key]
                else:
                    self._costs[type_set_data][key] = metric.get_costs(type_set_data)[key]
            for key in metric.get_scores(type_set_data):
                if key in self._scores[type_set_data]:
                    self._scores[type_set_data][key] += metric.get_scores(type_set_data)[key]
                else:
                    self._scores[type_set_data][key] = metric.get_scores(type_set_data)[key]

    @staticmethod
    def add_point(list_points, x, y, _type, name):
        """ Add point a list.

        Parameters
        ----------
        list_points : list[DataPlot]
            List of points.

        x : float
            Point axis x.

        y : float
            Point axis y.

        _type : str
            Type of data.

        name : str
            Plot's name.
        """
        if len(list_points) <= 0:
            list_points.append(DataPlot(name=name, _type=_type))

        list_points[0].add_point(x, y)

    @staticmethod
    def plot(ax, dps, label_prefix='', label=None):
        """ Generate plot.

        Parameters
        ----------
        ax
            Handle subplot.

        dps : numpy.array or list
            List of DataPlots.

        label_prefix : str
            This string is concatenate with title plot.

        label : str
            This string is the principal text in title plot.
        """
        # Get average plots
        if len(dps) > 0:
            if label is None:
                label = dps[0].get_name()
            x, y = BaseMetrics._get_data_per_col(dps)
            _x = x[:, 0]
            _y = np.nanmean(y, axis=1)
            _y_std = np.nanstd(y, axis=1)
            p = ax.plot(_x, _y, label='%s %s' % (label_prefix, label), lw=3)
            ax.fill_between(_x, _y - _y_std, _y + _y_std, alpha=0.1, color=p[0].get_color())

    @staticmethod
    def _get_data_per_col(dps):
        n = 0
        x = None
        y = None
        for dp in dps:
            x1, y1 = dp.get_data()
            x1 = np.array(x1, dtype=float)
            y1 = np.array(y1, dtype=float)
            if x1.ndim == 1:
                x1 = x1[:, np.newaxis]
                y1 = y1[:, np.newaxis]

            m = dp.len_data()
            if y is None:
                x = x1
                y = y1
            else:
                if m > n:
                    x = BaseMetrics._resize_rows(x, m)
                    y = BaseMetrics._resize_rows(y, m)
                elif m < n:
                    x1 = BaseMetrics._resize_rows(x1, n)
                    y1 = BaseMetrics._resize_rows(y1, n)

                x = np.hstack((x, x1))
                y = np.hstack((y, y1))

            n = max(n, m)
        return x, y

    @staticmethod
    def _resize_rows(a, nr):
        """ Resize rows in a array

        Parameters
        ----------
        a : numpy.array
            Array.

        nr : int
            New size of rows.

        Returns
        -------
        numpy.array
            Returns array with rows resize.
        """
        r = a.shape[0]
        c = 1
        if a.ndim > 1:
            c = a.shape[1]
        na = np.resize(a, (nr, c))
        if r < nr:
            na[r:nr, :] = np.NaN  # complete with nan

        return na


class EnsembleMetrics(BaseMetrics):
    """ Class for generate different metrics for ensemble models.

    Attributes
    ----------
    _models_metric : dict[BaseMetrics]
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
        self._models_metric = {}
        self._y_true_per_model = []
        self._y_pred_per_model = {}

    def get_models_metric(self):
        return self._models_metric

    def append_data(self, data, epoch, type_set_data):
        """ Append metrics data of ensemble.

        Parameters
        ----------
        data : list or array
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
        n = super(EnsembleMetrics, self).append_data(data, epoch, type_set_data=type_set_data)

        if len(data) > n:
            labels = self._model.get_result_labels()

            for model in self._model.get_models():
                s_model = model.get_name()
                if s_model not in self._models_metric:
                    self._models_metric[s_model] = FactoryMetrics().get_metric(model)

                n += 1
                BaseMetrics.add_point(self._models_metric[s_model].get_cost(type_set_data), epoch, data[n],
                                      labels[n], model.get_name())

                n = BaseMetrics.add_data(labels, model.get_name(),
                                         self._models_metric[s_model].get_costs(type_set_data),
                                         len(model.get_costs()), n, data, epoch)

                n = BaseMetrics.add_data(labels, model.get_name(),
                                         self._models_metric[s_model].get_scores(type_set_data),
                                         len(model.get_scores()), n, data, epoch)

        return n

    def append_prediction_per_model(self, _input, _target):
        """ Append prediction for each model in ensemble.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        Returns
        -------
        None
        """
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
            for name_model in metric._models_metric:
                if name_model in self._models_metric:
                    self._models_metric[name_model].append_metric(metric._models_metric[name_model])
                else:
                    self._models_metric[name_model] = metric._models_metric[name_model]

            super(EnsembleMetrics, self).append_metric(metric)
        else:
            if metric._model.get_name() in self._models_metric:
                self._models_metric[metric._model.get_name()].append_metric(metric)
            else:
                self._models_metric[metric._model.get_name()] = metric

    def plot_costs(self, max_epoch, title='Cost', log_xscale=False, log_yscale=False):
        """ Generate costs plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        title : str
            Plot title of cost.

        log_xscale : bool
            Flag for show plot x-axis in logarithmic scale.

        log_yscale : bool
            Flag for show plot y-axis in logarithmic scale.
        """
        costs = {'train': {}, 'test': {}}
        for name_model in self._models_metric:
            model_metric = self._models_metric[name_model]
            for type_set_data in ['train', 'test']:
                for key in model_metric.get_costs(type_set_data):
                    if key in self._costs[type_set_data]:
                        costs[type_set_data][key] += model_metric.get_costs(type_set_data)[key]
                    else:
                        costs[type_set_data][key] = model_metric.get_costs(type_set_data)[key]

        data_train = costs['train']
        data_test = costs['test']

        data = set(data_train) & set(data_test)
        f, _ = plt.subplots()
        N = len(data)
        cols = max(N // 2, 1)
        rows = max(N // cols, 1)
        for j, i in enumerate(data_train):
            ax = plt.subplot(rows, cols, j + 1)
            plt.hold(True)
            BaseMetrics.plot(ax, data_train[i], 'Train', self._model.get_name())
            BaseMetrics.plot(ax, data_test[i], 'Test', self._model.get_name())
            plt.hold(False)

            # noinspection PyTypeChecker
            ax.set_title('%s: %s' % (title, data_train[i][0].get_type()))

            if log_xscale:
                ax.set_xscale('log')
            if log_yscale:
                ax.set_yscale('log')
            ax.legend(loc='best')
            ax.set_xlim([0, max_epoch])
            plt.xlabel('epoch')
            plt.tight_layout()

        return f
