import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

import matplotlib
#matplotlib.use('TkAgg')

from theano import config, shared
from deepensemble.metrics import *
from deepensemble.utils import *
from deepensemble.utils.utils_test import get_mean_score, plot_graph2

# import matplotlib.pylab as plt
from sklearn import model_selection

import matplotlib.pyplot as plt

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
config.gcc.cxxflags = "-fbracket-depth=16000"
# config.exception_verbosity='high'
# config.compute_test_value='warn'

#############################################################################################################
# Load Data
#############################################################################################################
name_data = 'australian_scale'
data_input, data_target, classes_labels, name_db, desc, col_names = \
    load_data(name_data, data_home='../data', normalize=False)

#############################################################################################################
# Define Parameters nets
#############################################################################################################

n_features = data_input.shape[1]
n_classes = len(classes_labels)

n_output = n_classes
n_inputs = n_features

fn_activation = ActivationFunctions.sigmoid

n_ensemble_models = 4
n_neurons_model = (n_output + n_inputs) // 3

update_fn = sgd
name_update = 'SGD'

#############################################################################################################
# Define Parameters training
#############################################################################################################

# 10-Cross Validation, 300 epoch and 40 size mini-batch
args_train = {'max_epoch': 300, 'batch_size': 50, 'early_stop': False,
              'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True,
              'criterion_update_params': 'cost', 'maximization_criterion': True}

# ==========< Ensemble CIP  >================================================================================
s_beta = shared(np.cast['float32'](0))
s_lambda = shared(np.cast['float32'](0))
s_sigma = shared(np.cast['float32'](0))

ensemble = get_ensembleCIP_model(name='Ensamble CIP CS',
                                 n_input=n_features, n_output=n_output,
                                 n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                 classification=True,
                                 classes_labels=classes_labels,
                                 fn_activation1=fn_activation, fn_activation2=fn_activation,
                                 dist='CS',
                                 # lsp=1.5, lsm=0.5,
                                 bias_layer=False, mse_first_epoch=True, annealing_enable=False,
                                 update=update_fn, name_update=name_update,
                                 beta=s_beta, lamb=s_lambda, s=s_sigma, lr=0.005,
                                 params_update={'learning_rate': -0.05})

ensemble.compile(fast=True)
default_params_ensemble = ensemble.save_params()


def get_ensemble_cip(_name, _beta, _lamb, _s):
    ensemble.set_name(_name)
    ensemble.load_params(default_params_ensemble)
    s_beta.set_value(np.cast['float32'](_beta))
    s_lambda.set_value(np.cast['float32'](_lamb))
    s_sigma.set_value(np.cast['float32'](_s))

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################

y = get_index_label_classes(translate_target(data_target, classes_labels))
silverman = ITLFunctions.silverman(np.array(y)).eval()

ss = silverman * np.array([0.01, 0.1, 1, 5, 10, 20])
beta = [-1.0, -0.5, -0.3, 0, 0.3, 0.5, 1.0]
lamb = [-1.0, -0.5, -0.3, 0, 0.3, 0.5, 1.0]

bb, ll, sss = np.meshgrid(beta, lamb, ss)
parameters = list(zip(bb.flatten(), ll.flatten(), sss.flatten()))

path_data = 'test_params_cip/%s/' % name_db

name = 'Ensemble CIP'
scores = {}
diversity = {}

if not os.path.exists(path_data):
    os.makedirs(path_data)

Logger().log('Processing %s' % name)

Logger().reset()
models = []
folds = 5
test_size = 0.1

for b, l, s in Logger().progressbar(it=parameters, end='Finish'):
    # gets name ensemble
    name_model = name + ' %.2g %.2g %.2g' % (b, l, s)
    # make dirs
    _dir = path_data + name_model + '/'
    make_dirs(_dir)
    # generate ensemble
    file_data = _dir + 'data.pkl'

    if not os.path.exists(file_data):

        Logger().log_disable()

        _model = get_ensemble_cip(_name=name_model, _beta=b, _lamb=l, _s=s)

        m_scores = []
        m_diversity = []

        for fold in range(folds):

            Logger().log('Fold: %d' % (fold + 1))
            file_sets_fold = path_data + 'sets_fold_%d.pkl' % fold
            if not os.path.exists(file_sets_fold):
                # Generate testing and training sets
                input_train, input_test, target_train, target_test = \
                    model_selection.train_test_split(data_input, data_target, test_size=test_size)
                sets_data = Serializable((input_train, input_test, target_train, target_test))
                sets_data.save(file_sets_fold)
            else:
                # Load sets
                Logger().log('Load sets: %s' % file_sets_fold)
                sets_data = Serializable()
                sets_data.load(file_sets_fold)
                input_train, input_test, target_train, target_test = sets_data.get_data()

            file_model_fold = _dir + 'data_fold_%d.pkl' % fold
            score_train, score_test, metric = get_scores(_model, file_model_fold,
                                                    input_train, target_train, input_test, target_test, save=False, **args_train)

            m_scores.append((score_train, score_test))

            m_diversity.append(metric.get_diversity(metric=generalized_diversity)[0])

        data = {'scores': m_scores, 'diversity': m_diversity}
        s_data = Serializable(data)
        s_data.save(file_data)

        Logger().log_enable()
    else:
        Logger().log('Load file: %s' % file_data)
        s_data = Serializable()
        s_data.load(file_data)
        data = s_data.get_data()

    scores[(b, l, s)] = data['scores']
    diversity[(b, l, s)] = data['diversity']

fig, axes = plt.subplots(nrows=3, ncols=2)
fig.patch.set_visible(False)
for s1, ax in zip(ss, axes.flat):
    X, Y, Z = get_mean_score(scores, s1, True)
    ks = (s1 / silverman)
    if ks == 1:
        s_title = r'$\sigma=\sigma_s$'
    else:
        s_title = r'$\sigma=%.4g\sigma_s$' % (s1 / silverman)
    p = plot_graph2(ax, X, Y, Z, r'$\beta$', r'$\lambda$', r'Precisión', s_title)

plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
# noinspection PyUnboundLocalVariable
fig.colorbar(p, cax=cbar_ax, ax=axes.ravel().tolist())


fig, ax = plt.subplots()
fig.patch.set_visible(False)
X, Y, Z = get_mean_score(scores, silverman, False)
s_title = 'Precisión\n$\sigma=\sigma_s$'
p = plot_graph2(ax, X, Y, Z, r'$\beta$', r'$\lambda$', r'Precisión', s_title)
plt.tight_layout()
fig.colorbar(p, ax=ax)

fig, axes = plt.subplots(nrows=3, ncols=2)
fig.patch.set_visible(False)
for s1, ax in zip(ss, axes.flat):
    X, Y, Z = get_mean_diversity(diversity, s1)
    ks = (s1 / silverman)
    if ks == 1:
        s_title = r'$\sigma=\sigma_s$'
    else:
        s_title = r'$\sigma=%.4g\sigma_s$' % (s1 / silverman)
    p = plot_graph2(ax, X, Y, Z, r'$\beta$', r'$\lambda$', r'Diversidad', s_title)

plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
# noinspection PyUnboundLocalVariable
fig.colorbar(p, cax=cbar_ax, ax=axes.ravel().tolist())

fig, ax = plt.subplots()
fig.patch.set_visible(False)
X, Y, Z = get_mean_diversity(diversity, silverman)
s_title = 'Diversidad Generalizada\n$\sigma=\sigma_s$'
p = plot_graph2(ax, X, Y, Z, r'$\beta$', r'$\lambda$', r'Diversidad', s_title)
plt.tight_layout()
fig.colorbar(p, ax=ax)

plt.show()
