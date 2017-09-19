import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.metrics import accuracy_score
from .serializable import Serializable
from ..utils.utils_translation import TextTranslation

from .singleton import Singleton

__all__ = ['ConfigPlot', 'DataPlot', 'add_data', 'add_point', 'plot', 'plot_data', 'plot_list_data', 'plot_data_training_ensemble']

class ConfigPlot(Singleton):

    def __init__(self):
        self._fig_size = (7, 5)
        self._dpi = 80
        self._style = 'fivethirtyeight'
        self._hold = True
        self.loc = 'best'
        self.size_font = 10

        matplotlib.rcParams.update({'font.size': self.size_font})
        plt.style.use(self._style)

    def set_fig_size(self, fig_size):
        self._fig_size = fig_size

    def set_dpi(self, dpi):
        self._dpi = dpi

    def set_style(self, style):
        self._style = style
        plt.style.use(self._style)

    def set_hold(self, hold):
        self._hold = hold

    def set_loc(self, loc):
        self._loc = loc

    def set_size_font(self, size):
        self.size_font = size
        matplotlib.rcParams.update({'font.size': size})

    def get_fig_size(self):
        return self._fig_size

    def get_dpi(self):
        return self._dpi

    def get_style(self):
        return self._style

    def get_hold(self):
        return self._hold

    def get_loc(self):
        return self.loc


class DataPlot(Serializable):
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
        super(DataPlot, self).__init__()
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

    def set_name(self, name):
        """ Setter name of plot.

        Parameters
        ----------
        name : str
            Name of plot (using as label plot).
        """
        self.__name = name

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


def add_data(labels, model_name, data_dict, n_data, index, data, epoch):
    """ Appends data training in data_dict (dictionary).

    Parameters
    ----------
    labels : list[]
        List of label data.

    model_name : str
        Name of model.

    data_dict : dict
        Dictionary of data.

    n_data : int
        Number of data.

    index : int
        Index of data that it wants to append.

    data : list[]
        Source of data.

    epoch : int
        Current epoch training.

    Returns
    -------
    int
        Returns current index (index = index + n_data)
    """
    for _ in range(n_data):
        index += 1
        label = labels[index]
        if label not in data_dict:
            data_dict[label] = []
        add_point(data_dict[label], epoch, data[index], label, model_name)
    return index


def plot_data(ax, list_data_plots, x_max, x_min=0.0, title=None, log_xscale=False, log_yscale=False):
    """ Plot list data plots.

    Parameters
    ----------
    ax
        Handler plot.

    list_data_plots : tuple(list[DataPlot], str)
        List DataPlots.

    x_max : float
        Limit x-axis plot.

    x_min : float
        Limit x-axis plot.

    title : str
        Title of plot.

    log_xscale : bool
        Flag for scaling x-axis plot.

    log_yscale
        Flag for scaling y-axis plot.
    """
    if title is None:
        title = TextTranslation().get_str('Cost')
    else:
        plt.ylabel(title)

    linestyles = ['-', '--', '-.', ':']
    for i, (data_plot, prefix) in enumerate(list_data_plots):
        plot(ax, data_plot, prefix, linestyle=linestyles[(i + 4) % 4])

    ax.set_title(title)
    if log_xscale:
        ax.set_xscale('log')
    if log_yscale:
        ax.set_yscale('log')
    ax.legend(loc=ConfigPlot().get_loc())
    ax.set_xlim([x_min, x_max])
    plt.xlabel(TextTranslation().get_str('epoch'))
    plt.tight_layout()


def plot_list_data(list_data, x_max, title=None, log_xscale=False, log_yscale=False):
    """ Generate plot of list data.

    Parameters
    ----------
    list_data : list[]
        List of data to plot where the structure for each elements is as follows:

        tuple: (list[DataPlots], label_plot)

    x_max : float
        Limit x-axis plot.

    title : str
        Plot title of cost.

    log_xscale : bool
        Flag for show plot x-axis in logarithmic scale.

    log_yscale : bool
        Flag for show plot y-axis in logarithmic scale.
    """
    if title is None:
        title = TextTranslation().get_str('Cost')

    N = len(list_data)

    if N > 0:
        f = None
        if ConfigPlot().get_hold():
            f = plt.figure(figsize=ConfigPlot().get_fig_size(), dpi=ConfigPlot().get_dpi())

        cols = max(N // 2, 1)
        rows = max(N // cols, 1)
        for j, (data, _type) in enumerate(list_data):
            if (rows * cols == 1) or not ConfigPlot().get_hold():
                plt.figure(figsize=ConfigPlot().get_fig_size(), dpi=ConfigPlot().get_dpi())
                ax = plt.gca()
            else:
                ax = plt.subplot(rows, cols, j + 1)
            str_title = '%s: %s' % (title, _type) if title != '' else _type
            plot_data(ax, data, x_max=x_max,
                      title=str_title,
                      log_xscale=log_xscale,
                      log_yscale=log_yscale)

        return f

    else:
        return None


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


def plot(ax, dps, label_prefix='', label=None, linestyle='-'):
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

    linestyle: str
        Style of plot line.
    """
    # Get average plots
    if len(dps) > 0:
        if label is None:
            label = dps[0].get_name()
        x, y = _get_data_per_col(dps)
        y = np.squeeze(y)
        if y.ndim <= 1:
            y = y[:, np.newaxis]
        elif y.ndim > 2:
            y = np.mean(y, axis=-1)
        _x = x[:, 0]
        _y = np.nanmean(y, axis=-1)
        _y_std = np.nanstd(y, axis=-1)
        ax.plot(_x, _y, label='%s %s' % (label_prefix, label), lw=3, linestyle=linestyle)
        ax.errorbar(_x, _y, _y_std, linestyle='None', marker='o', lw=2, ecolor='k', capsize=5, color='k')


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
                x = _resize_rows(x, m)
                y = _resize_rows(y, m)
            elif m < n:
                x1 = _resize_rows(x1, n)
                y1 = _resize_rows(y1, n)

            x = np.hstack((x, x1))
            y = np.hstack((y, y1))

        n = max(n, m)
    return x, y


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

def plot_data_training_ensemble(ensemble, max_epoch, input_train, input_test, target_train, target_test, metrics):

    from deepensemble.utils import plot_pdf

    plt.style.use(ConfigPlot().get_style())

    f = plt.figure(figsize=ConfigPlot().get_fig_size(), dpi=ConfigPlot().get_dpi())

    e_train = ensemble.error(input_train, ensemble.translate_target(target_train)).eval()
    e_test = ensemble.error(input_test, ensemble.translate_target(target_test)).eval()

    ax = plt.subplot(2, 1, 1)
    for i in range(ensemble.get_fan_out()):
        plot_pdf(ax, e_test[:, i], label=TextTranslation().get_str('output') + ' %d' % (i + 1), x_min=-3, x_max=3, n_points=1000)
    plt.legend()

    ax = plt.subplot(2, 1, 2)
    for i in range(ensemble.get_fan_out()):
        plot_pdf(ax, e_train[:, i], label=TextTranslation().get_str('output') + ' %d' % (i + 1), x_min=-3, x_max=3, n_points=1000)
    plt.legend()

    # noinspection PyRedeclaration
    f = plt.figure(figsize=ConfigPlot().get_fig_size(), dpi=ConfigPlot().get_dpi())
    msg_train = ''
    msg_test = ''
    row = math.ceil(ensemble.get_num_models() / 2.0)
    col = 2
    for i, model in enumerate(ensemble.get_models()):
        e_test = model.error(input_test, model.translate_target(target_test)).eval()

        ax = plt.subplot(row, col, i + 1)
        for j in range(ensemble.get_fan_out()):
            plot_pdf(ax, e_test[:, j], label='Test output %d' % (j + 1), x_min=-3, x_max=3, n_points=1000)
        plt.legend()
        plt.title('Model %s' % model.get_name())

        pred_test = model.predict(input_test)
        pred_train = model.predict(input_train)

        msg_test += '%s %s test: %.4g\n' % \
                    (TextTranslation().get_str('Accuracy_model'),
                     model.get_name(), accuracy_score(pred_test, target_test))
        msg_train += '%s %s train: %.4g\n' % \
                     (TextTranslation().get_str('Accuracy_model'),
                      model.get_name(), accuracy_score(pred_train, target_train))

    print(msg_test)
    print('%s test: %.4g' % (TextTranslation().get_str('Accuracy_ensemble'),
                                            accuracy_score(ensemble.predict(input_test), target_test)))
    print('--' * 10)
    print(msg_train)
    print('%s train: %.4g' % (TextTranslation().get_str('Accuracy_ensemble'),
                                             accuracy_score(ensemble.predict(input_train), target_train)))

    plt.tight_layout()

    metrics.plot_cost(max_epoch=max_epoch, title=TextTranslation().get_str('Cost_CIP'))
    metrics.plot_costs(max_epoch=max_epoch, title=TextTranslation().get_str('Costs_CIP'))
    metrics.plot_scores(max_epoch=max_epoch, title=TextTranslation().get_str('Accuracy_CIP'))

    metrics.append_prediction(input_test, target_test, append_last_pred=True)
    metrics.plot_confusion_matrix()

    plt.show()

