from .serializable import Serializable
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['DataPlot', 'add_data', 'add_point', 'plot', 'plot_data', 'plot_list_data']


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


def plot_data(ax, list_data_plots, x_max, x_min=0.0, title='Cost', log_xscale=False, log_yscale=False):
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
    plt.hold(True)

    for data_plot, prefix in list_data_plots:
        plot(ax, data_plot, prefix)

    # plt.hold(False)

    ax.set_title(title)
    if log_xscale:
        ax.set_xscale('log')
    if log_yscale:
        ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlim([x_min, x_max])
    plt.xlabel('epoch')
    plt.tight_layout()


def plot_list_data(list_data, x_max, title='Cost', log_xscale=False, log_yscale=False):
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
    N = len(list_data)

    if N > 0:
        f, _ = plt.subplots()

        cols = max(N // 2, 1)
        rows = max(N // cols, 1)
        for j, (data, _type) in enumerate(list_data):
            ax = plt.subplot(rows, cols, j + 1)
            plot_data(ax, data, x_max=x_max,
                      title='%s: %s' % (title, _type),
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
        x, y = _get_data_per_col(dps)
        y = np.squeeze(y)
        if y.ndim <= 1:
            y = y[:,np.newaxis]
        _x = x[:, 0]
        _y = np.nanmean(y, axis=-1)
        _y_std = np.nanstd(y, axis=-1)
        p = ax.plot(_x, _y, label='%s %s' % (label_prefix, label), lw=3)

        yn = _y - _y_std
        yp = _y + _y_std

        ax.fill_between(_x, yn, yp, alpha=0.1, color=p[0].get_color())


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
