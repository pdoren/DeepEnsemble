import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

from theano import config
from deepensemble.utils import *

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
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale', data_home='../data')

input_train, input_test, target_train, target_test = \
    model_selection.train_test_split(data_input, data_target, test_size=0.3)

#############################################################################################################
# Define Parameters nets
#############################################################################################################

n_features = input_train.shape[1]
n_classes = len(classes_labels)

n_output = n_classes
n_inputs = n_features

fn_activation = ActivationFunctions.sigmoid

n_neurons = (n_inputs + n_output) // 2

lr = 0.01
reg_l1 = 0.0001
reg_l2 = 0.0001

#############################################################################################################
# Define Parameters training
#############################################################################################################

max_epoch = 300
folds = 5
batch_size = 50
training = True

args_train_default = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': False,
              'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True}

args_train_cip = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': False,
              'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True,
              'criterion_update_params': 'cost', 'maximization_criterion': True}

y = get_index_label_classes(translate_target(data_target, classes_labels))
silverman = ITLFunctions.silverman(np.array(y)).eval()

update_fn = sgd
name_update = 'SGD'


# ==========< Ensemble   >===================================================================================
def get_ensemble_ncl(_name, _n_neurons, fast=True):
    ensemble = get_ensembleNCL_model(name=_name, classification=True, classes_labels=classes_labels,
                                     n_input=n_features, n_output=n_output,
                                     n_ensemble_models=4, n_neurons_models=_n_neurons,
                                     fn_activation1=fn_activation, fn_activation2=fn_activation,
                                     update=update_fn, name_update=name_update,
                                     lamb=0.3, params_update={'learning_rate': 0.03}
                                     )
    ensemble.compile(fast=fast)

    return ensemble


# noinspection PyUnusedLocal
def get_ensemble_cip(_name, _n_neurons, fast=True):
    ensemble = get_ensembleCIP_model(name=_name, classification=True, classes_labels=classes_labels,
                                     n_input=n_features, n_output=n_output,
                                     n_ensemble_models=4, n_neurons_models=_n_neurons,
                                     is_cip_full=False,
                                     fn_activation1=fn_activation, fn_activation2=fn_activation,
                                     dist='CS',
                                     beta=0.1, lamb=0.5, s=silverman,
                                     lsp=1.5, lsm=0.5,
                                     lr=0.005,
                                     bias_layer=False, mse_first_epoch=True, annealing_enable=True,
                                     update=update_fn, name_update=name_update,
                                     params_update={'learning_rate': -0.03})
    ensemble.compile(fast=fast)

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################
parameters = [n for n in range(2, 2 * n_neurons, 2)]

list_ensemble = [(get_ensemble_ncl, args_train_default, 'Ensamble NCL'), (get_ensemble_cip, args_train_cip, 'Ensamble CIPL CS')]
path_data = 'test_ensemble_n_neurons/'

f, ax = plt.subplots()
f.patch.set_visible(False)

for get_ensemble, args_train, name in list_ensemble:

    data = {}
    scores = []

    if not os.path.exists(path_data):
        os.makedirs(path_data)
    Logger().log('Processing %s' % name)

    Logger().reset()
    for _p in Logger().progressbar(it=parameters, end='Finish'):
        file_data = path_data + '%s_%s_%d.pkl' % (name, name_db, _p)
        if not os.path.exists(file_data):

                Logger().log_disable()
                model = get_ensemble(_name=name, _n_neurons=_p)

                metrics, best_score, list_score = test_model(cls=model,
                                                             input_train=input_train, target_train=target_train,
                                                             input_test=input_test, target_test=target_test,
                                                             folds=folds, **args_train)

                Logger().log_enable()

                s_data = Serializable((list_score, _p, best_score))
                s_data.save(file_data)
        else:
            Logger().log('Load file: %s' % file_data)
            s_data = Serializable()
            s_data.load(file_data)
            list_score, p, best_score = s_data.get_data()

        data[_p] = list_score
        scores.append(best_score)

    scores = np.array(scores)

    #############################################################################################################
    #
    #  PLOT DATA CLASSIFIERS
    #
    #############################################################################################################

    list_dp = []
    for i in range(folds):
        y = list([data[l][i][0] for l in parameters])
        x = list(np.array(parameters) / n_inputs)
        dp = DataPlot(name=name, _type='score')
        dp.set_data(x, y)
        list_dp.append(dp)

    plot_data(ax, [(list_dp, 'Precisión')],
              x_max=(max(parameters) + 1) / n_inputs, x_min=(min(parameters) - 1) / n_inputs,
              title='Precisión por neuronas\n(Conjunto de Entrenamiento)')

plt.legend()
plt.xlabel('$n^o$ neuronas / $n^o$ caracteristicas')
plt.ylabel('Precisión')
plt.tight_layout()

plt.show()
