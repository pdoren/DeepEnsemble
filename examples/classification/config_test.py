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

def plot_pdf_error(model, _input, _target, label_plot, ax, fig, n_points=500, xmin=-1.5, xmax=1.5, lim_y=0.05):
    error = model.error(_input, model.translate_target(_target)).eval()
    N = len(error)
    s = 1.06 * np.std(error) / np.power(N, 0.2)  # Silverman

    x_plot = np.linspace(xmin, xmax, n_points)

    linestyles = ['-', '--', '-.', ':']
    L = model.get_fan_out()
    for j in range(L):
        kde = KernelDensity(kernel='gaussian', bandwidth=s)
        e = error[:, j]
        kde.fit(e[:, np.newaxis])
        y = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
        ax.plot(x_plot, y / np.sum(y), linestyle=linestyles[(j + 4) % 4], label='Salida %d' % (j + 1))

    plt.legend()
    plt.title(label_plot)
    plt.xlabel('Error')
    plt.ylabel('PDF del error')
    ax.set_ylim([0, lim_y])
    plt.tight_layout()