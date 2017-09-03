import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

from theano import config
from deepensemble.utils import *

import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

plt.style.use('fivethirtyeight')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 3

ConfigPlot().set_size_font(10)
ConfigPlot().set_dpi(80)
ConfigPlot().set_fig_size((7, 5))
ConfigPlot().set_hold(False)

from sklearn.neighbors.kde import KernelDensity
from pylab import *

config.optimizer = 'fast_compile'
# config.exception_verbosity='high'
# config.compute_test_value='warn'

def plot_pdf_error(models, _input, _target, label_plot, ax, fig, n_points=500, xmin=-3, xmax=3, lim_y=0.05, title=''):

    linestyles = ['-', '--', '-.', ':']
    for j, model in enumerate(models):
        error = model.error(_input, _target).eval()
        N = len(error)
        s = 1.06 * np.std(error) / np.power(N, 0.2)  # Silverman

        x_plot = np.linspace(xmin, xmax, n_points)

        kde = KernelDensity(kernel='gaussian', bandwidth=s)
        kde.fit(error)
        y = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
        if title != '':
            ax.plot(x_plot, y / np.sum(y), linestyle=linestyles[(j + 4) % 4], label=title)
        else:
            ax.plot(x_plot, y / np.sum(y), linestyle=linestyles[(j + 4) % 4], label=model.get_name())

    plt.legend()
    plt.title(label_plot)
    plt.xlabel('Error')
    plt.ylabel('PDF del error')
    ax.set_ylim([0, lim_y])
    plt.tight_layout()

def plot_pdf_error_1(model, _input, _target, label_plot, ax, fig, n_points=500, xmin=-3, xmax=3, lim_y=0.05, title='', linestyle='-'):

    error = model.error(_input, _target).eval()
    N = len(error)
    s = 1.06 * np.std(error) / np.power(N, 0.2)  # Silverman

    x_plot = np.linspace(xmin, xmax, n_points)

    kde = KernelDensity(kernel='gaussian', bandwidth=s)
    kde.fit(error)
    y = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
    if title != '':
        ax.plot(x_plot, y / np.sum(y), linestyle=linestyle, label=title)
    else:
        ax.plot(x_plot, y / np.sum(y), linestyle=linestyle, label=model.get_name())

    plt.legend()
    plt.title(label_plot)
    plt.xlabel('Error')
    plt.ylabel('PDF del error')
    ax.set_ylim([0, lim_y])
    plt.tight_layout()
