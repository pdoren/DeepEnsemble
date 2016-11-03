import math
import os

import matplotlib.pylab as plt
import numpy as np
from sklearn import cross_validation

from deepensemble.combiner import *
from deepensemble.layers import *
from deepensemble.models import *
from deepensemble.utils import *
from deepensemble.utils.utils_functions import ActivationFunctions

SEED = 13
plt.style.use('ggplot')

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale', data_home='../data')

input_train, input_test, target_train, target_test = \
    cross_validation.train_test_split(data_input, data_target, test_size=0.3, stratify=data_target,
                                      random_state=SEED)

#############################################################################################################
# Define Parameters nets
#############################################################################################################

n_features = input_train.shape[1]
n_classes = len(classes_labels)

n_output = n_classes
n_inputs = n_features

fn_activation = ActivationFunctions.sigmoid

n_ensemble_models = 4
n_neurons = n_features * 2


lr = 0.05
reg_l1 = 0.0001
reg_l2 = 0.0001

#############################################################################################################
# Define Parameters training
#############################################################################################################

max_epoch = 300
folds = 5
batch_size = 32
training = True

args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': True,
              'improvement_threshold': 0.9995, 'update_sets': True}


# ==========< Ensemble   >===================================================================================
def get_ensemble_ncl(_name, _n_neurons, n_models, fast=True):
    ensemble = EnsembleModel(name=_name)
    for _i in range(n_models):
        net = Sequential("net%d" % (_i + 1), "classifier", classes_labels)
        net.add_layer(Dense(n_input=n_inputs, n_output=_n_neurons,
                            activation=fn_activation))
        net.add_layer(Dense(n_output=n_output, activation=fn_activation))
        net.append_cost(mse, name="MSE")
        net.set_update(sgd, name="SGD", learning_rate=lr)
        ensemble.append_model(net)

    ensemble.add_cost_ensemble(fun_cost=neg_corr, name="NEG_CORR", lamb_neg_corr=0.6)
    ensemble.set_combiner(PluralityVotingCombiner())
    ensemble.compile(fast=fast)

    return ensemble


def get_ensemble_ckl(_name, _n_neurons, n_models, fast=True):
    ensemble = EnsembleModel(name=_name)
    for _i in range(n_models):
        net = Sequential("net%d" % (_i + 1), "classifier", classes_labels)
        net.add_layer(Dense(n_input=n_inputs, n_output=_n_neurons,
                            activation=fn_activation))
        net.add_layer(Dense(n_output=n_output, activation=fn_activation))
        net.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
        net.set_update(sgd, name="SGD", learning_rate=lr)
        ensemble.append_model(net)

    ensemble.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.6)
    ensemble.set_combiner(PluralityVotingCombiner())
    ensemble.compile(fast=fast)

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################
xx = np.linspace(1, n_neurons, 7, dtype='int')
yy = np.linspace(1, 21, 7, dtype='int')
dd = np.array(np.meshgrid(xx, yy))
parameters = [(p1, p2) for p1, p2 in zip(dd[0, :].flatten(), dd[1, :].flatten())]

list_ensemble = [(get_ensemble_ncl, 'Ensemble NCL'), (get_ensemble_ckl, 'Ensemble KLG Correntropy')]
path_data = 'test_ensemble_n_neurons_vs_models/'

for get_ensemble, name in list_ensemble:

    data = {}
    scores = []

    if not os.path.exists(path_data):
        os.makedirs(path_data)
    Logger().log('Processing %s' % name)
    file_data = path_data + '%s_%s.pkl' % (name, name_db)

    if not os.path.exists(file_data):
        Logger().reset()
        for _p1, _p2 in Logger().progressbar(it=parameters, end='Finish'):
            Logger().log_disable()
            model = get_ensemble(_name=name, _n_neurons=_p1, n_models=_p2)

            metrics, best_score, list_score = test_model(cls=model,
                                                         input_train=input_train, target_train=target_train,
                                                         input_test=input_test, target_test=target_test,
                                                         folds=folds, **args_train)

            scores.append(best_score)
            data[(_p1, _p2)] = {'model': model, 'metrics': metrics, 'list_score': list_score}
            Logger().log_enable()

        scores = np.array(scores)
        s_data = Serializable((data, parameters, scores))
        s_data.save(file_data)
    else:
        Logger().log('Load file: %s' % file_data)
        s_data = Serializable()
        s_data.load(file_data)
        data, parameters, scores = s_data.get_data()

    #############################################################################################################
    #
    #  PLOT DATA CLASSIFIERS
    #
    #############################################################################################################

    f = plt.figure()
    ax = f.gca(projection='3d')
    plt.hold(True)
    list_dp = []


    z = [np.mean(data[(l1, l2)]['list_score']) for l1, l2 in parameters]
    s = [np.std(data[(l1, l2)]['list_score']) for l1, l2 in parameters]
    x = np.array(xx) / n_features
    y = np.array(yy) / n_features

    cm = plt.cm.get_cmap('RdYlBu')
    # sc = plt.scatter(x, y, c=z, vmin=0, vmax=1, s=s, cmap=cm)
    sc = ax.plot_surface(x, y, z, cmap=cm)
    plt.colorbar(sc)

    plt.xlabel('$n^o neurons / n^o features$')
    plt.ylabel('$n^o models / n^o features$')
    plt.tight_layout()

plt.show()
