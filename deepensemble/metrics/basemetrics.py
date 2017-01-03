import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pylab as plt
from collections import OrderedDict
from ..utils.serializable import Serializable
from ..utils.utils_plot import add_point, add_data, plot_data, plot_list_data

__all__ = ['BaseMetrics', 'EnsembleMetrics', 'FactoryMetrics']


def concatenate_data(x, y):
    """ Concatenate 2 arrays """
    x = x[:, np.newaxis] if len(x.shape) == 1 else x
    y = y[:, np.newaxis] if len(y.shape) == 1 else y

    return np.vstack((x, y))


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


class BaseMetrics(Serializable):
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

    _y_true : list[numpy.array]
        List of array with target samples.

    _y_pred : list[numpy.array]
        List of array with output or prediction of model.

    Parameters
    ----------
    model : Model
        Model.
    """

    def __init__(self, model):
        super(BaseMetrics, self).__init__()
        self._model = model
        self._error = {'train': [], 'test': []}
        self._cost = {'train': [], 'test': []}
        self._costs = {'train': OrderedDict(), 'test': OrderedDict()}
        self._scores = {'train': OrderedDict(), 'test': OrderedDict()}
        self._y_pred = []
        self._y_true = []

    def get_model(self):
        """ Getter model.

        Returns
        -------
        Model
            Returns current model in metric.
        """
        return self._model

    def get_cost(self, type_set_data):
        """ Getter total cost.

        Parameters
        ----------
        type_set_data : str
            This string means what kind of data is passed: train or test.

        Returns
        -------
        list[]
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
        add_point(self._error[type_set_data], epoch, data[n], labels[n], self._model.get_name())

        n += 1  # data[1] is the total cost
        add_point(self._cost[type_set_data], epoch, data[n], labels[n], self._model.get_name())

        n_cost = len(self._model.get_costs())
        if n_cost > 1:
            n = add_data(labels, self._model.get_name(),
                         self.get_costs(type_set_data), n_cost, n, data, epoch)
        elif n_cost == 1 and not self._model.is_fast_compiled():
            n += 1

        n_data_score = 1 if self._model.is_fast_compiled() else len(self._model.get_scores())
        return add_data(labels, self._model.get_name(),
                        self.get_scores(type_set_data), n_data_score, n, data, epoch)

    def plot_cost(self, max_epoch, title='Cost', log_xscale=False, log_yscale=False):
        """ Generate cost plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        title : str
            Plot title of training cost.

        log_xscale : bool
            Flag for show plot x-axis in logarithmic scale.

        log_yscale : bool
            Flag for show plot y-axis in logarithmic scale.
        """

        data_train = self.get_cost('train')
        data_test = self.get_cost('test')

        if len(data_train) > 0:
            f, ax = plt.subplots()

            data = [(data_train, 'Train'), (data_test, 'Test')]

            plot_data(ax, data, x_max=max_epoch,
                      title=title,
                      log_xscale=log_xscale, log_yscale=log_yscale)

            return f
        else:
            return None

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
        list_data = []
        for key in self.get_costs('train'):
            data_train = self.get_costs('train')[key]
            data_test = self.get_costs('test')[key]
            list_data.append(([(data_train, 'Train'), (data_test, 'Test')], data_train[0].get_type()))

        return plot_list_data(list_data=list_data,
                              x_max=max_epoch, title=title, log_xscale=log_xscale, log_yscale=log_yscale)

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
        list_data = []
        for key in self.get_scores('train'):
            data_train = self.get_scores('train')[key]
            data_test = self.get_scores('test')[key]
            list_data.append(([(data_train, 'Train'), (data_test, 'Test')], data_train[0].get_type()))

        return plot_list_data(list_data=list_data,
                              x_max=max_epoch, title=title, log_xscale=log_xscale, log_yscale=log_yscale)

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

    def append_prediction(self, _input, _target, append_last_pred=False):
        """ Add a sample of prediction and target for generating metrics.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        append_last_pred : bool
            This flag indicates that the current prediction is append in the last saved prediction.

        Returns
        -------
        float
            Return model score (classifier: accuracy, regressor: MSE).
        """
        _output = self._model.predict(_input)
        n = len(self._y_pred)
        if append_last_pred:
            if n > 0:
                self._y_pred[n - 1] = concatenate_data(self._y_pred[n - 1], np.squeeze(_output))
                self._y_true[n - 1] = concatenate_data(self._y_true[n - 1], np.squeeze(_target))
            else:
                self._y_pred += [np.squeeze(_output)]
                self._y_true += [np.squeeze(_target)]

        return self.get_score_prediction(_target, _output)

    def get_score_prediction(self, _target, _prediction):
        """ Get score the prediction.

        Parameters
        ----------
        _target : numpy.array
            Target sample.

        _prediction : numpy.array
            Prediction of model.

        Returns
        -------
        float
            Returns prediction od model, in case of classifier model return accuracy and in case of regressor model
            return mean square error.
        """
        if self._model.is_classifier():
            return accuracy_score(np.squeeze(_target), np.squeeze(_prediction))
        else:
            return mean_squared_error(np.squeeze(_target), np.squeeze(_prediction))


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
        """ Gets Ensemble models.

        Returns
        -------
        list[]
            Returns list of Ensemble models.
        """
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

        if (len(data) - 1) > n:
            labels = self._model.get_result_labels()

            for model in self._model.get_models():
                s_model = model.get_name()
                if s_model not in self._models_metric:
                    self._models_metric[s_model] = FactoryMetrics().get_metric(model)

                n_costs = len(model.get_costs())
                if n_costs > 0:
                    n += 1
                add_point(self._models_metric[s_model].get_cost(type_set_data), epoch, data[n], labels[n],
                          model.get_name())

                n = add_data(labels=labels, model_name=model.get_name(),
                             data_dict=self._models_metric[s_model].get_costs(type_set_data),
                             n_data=n_costs, index=n, data=data, epoch=epoch)

                n = add_data(labels=labels, model_name=model.get_name(),
                             data_dict=self._models_metric[s_model].get_scores(type_set_data),
                             n_data=len(model.get_scores()), index=n, data=data, epoch=epoch)

        return n

    def append_prediction(self, _input, _target, append_last_pred=False):
        """ Append prediction (using for generate reports).

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target :  numpy.array
            Target sample.

        append_last_pred : bool
            This flag indicates that the current prediction is append in the last saved prediction.

        Returns
        -------
        float
            Returns the score prediction.
        """
        self.append_prediction_per_model(_input, _target, append_last_pred)
        return super(EnsembleMetrics, self).append_prediction(_input, _target, append_last_pred)

    def append_prediction_per_model(self, _input, _target, append_last_pred=False):
        """ Append prediction for each model in ensemble.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        append_last_pred : bool
            This flag indicates that the current prediction is append in the last saved prediction.

        Returns
        -------
        None
        """
        _target = np.squeeze(_target)
        n = len(self._y_true_per_model)
        if append_last_pred:
            if n > 0:
                self._y_true_per_model[n - 1] = concatenate_data(self._y_true_per_model[n - 1], _target)
            else:
                self._y_true_per_model += [_target]

        for _model in self._model.get_models():
            output = np.squeeze(_model.predict(_input))
            if _model.get_name() not in self._y_pred_per_model:
                self._y_pred_per_model[_model.get_name()] = []

            n = len(self._y_pred_per_model[_model.get_name()])
            if append_last_pred:
                if n > 0:
                    self._y_pred_per_model[_model.get_name()][n - 1] = concatenate_data(
                        self._y_pred_per_model[_model.get_name()][n - 1], output)
                else:
                    self._y_pred_per_model[_model.get_name()] += [output]

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

    def plot_costs(self, max_epoch, name=None, title='Cost', log_xscale=False, log_yscale=False):
        """ Generate costs plot.

        Parameters
        ----------
        max_epoch : int
            Number of epoch of training.

        name : str
            Name plot.

        title : str
            Plot title of cost.

        log_xscale : bool
            Flag for show plot x-axis in logarithmic scale.

        log_yscale : bool
            Flag for show plot y-axis in logarithmic scale.
        """
        if name is None:
            name = self._model.get_name()

        costs = {'train': {}, 'test': {}}
        for name_model in self._models_metric:
            model_metric = self._models_metric[name_model]
            for type_set_data in ['train', 'test']:
                for key in model_metric.get_costs(type_set_data):
                    cost = model_metric.get_costs(type_set_data)[key]
                    cost[0].set_name(name)
                    if key in self._costs[type_set_data]:
                        costs[type_set_data][key] += cost
                    else:
                        costs[type_set_data][key] = cost

        list_data = []
        for key in costs['train']:
            data_train = costs['train'][key]
            data_test = costs['test'][key]
            list_data.append(([(data_train, 'Train'), (data_test, 'Test')], key))

        return plot_list_data(list_data=list_data,
                              x_max=max_epoch, title=title, log_xscale=log_xscale, log_yscale=log_yscale)
