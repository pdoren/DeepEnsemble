import os

import matplotlib.pylab as plt
import numpy as np

from deepensemble.utils import *
from deepensemble.utils.utils_test import make_dirs
from deepensemble.utils.utils_functions import ActivationFunctions
from theano import shared
from sklearn import cross_validation
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata


SEED = 13
plt.style.use('ggplot')

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = \
    load_data('germannumer_scale', data_home='../data')

#############################################################################################################
# Define Parameters nets
#############################################################################################################

n_features = data_input.shape[1]
n_classes = len(classes_labels)

n_output = n_classes
n_inputs = n_features

fn_activation = ActivationFunctions.sigmoid

n_ensemble_models = 3
n_neurons = (n_output + n_inputs) // 2

n_neurons_models = n_neurons // n_ensemble_models

lr = 0.02

#############################################################################################################
# Define Parameters training
#############################################################################################################

# 10-Cross Validation, 300 epoch and 40 size mini-batch
args_train = {'max_epoch': 300, 'batch_size': 40, 'early_stop': True,
              'improvement_threshold': 0.995, 'update_sets': True}



# ==========< Ensemble CIP  >================================================================================
s_beta = shared(np.cast['float32'](0))
s_lambda = shared(np.cast['float32'](0))
s_sigma = shared(np.cast['float32'](0))

ensemble = ensembleCIP_classification(name='ensemble',
                                      n_feature=n_features, classes_labels=classes_labels,
                                      n_ensemble_models=n_ensemble_models,
                                      n_neurons_models=n_neurons_models,
                                      fn_activation1=ActivationFunctions.tanh,
                                      fn_activation2=ActivationFunctions.sigmoid,
                                      beta=s_beta, lamb=s_lambda, s=s_sigma, lr=lr)

ensemble.compile(fast=True)

def get_ensemble_cip(_name, _beta, _lamb, s):
    ensemble.set_name(_name)
    s_beta.set_value(np.cast['float32'](_beta))
    s_lambda.set_value(np.cast['float32'](_lamb))
    s_sigma.set_value(np.cast['float32'](s))

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################

y = get_index_label_classes(translate_target(data_target, classes_labels))
silverman = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()

ss = silverman * np.array([0.01, 0.1, 1, 5, 10, 20 ])
beta = [-1.0, -0.5, -0.2, 0, 0.1, 0.2, 0.4, 0.8, 1.0]
lamb = [-1.0, -0.5, -0.2, 0, 0.1, 0.2, 0.5, 1.0]

bb, ll, sss = np.meshgrid(beta, lamb, ss)
parameters = list(zip(bb.flatten(), ll.flatten(), sss.flatten()))

path_data = 'test_params_cip/%s/' % name_db

name = 'Ensemble CIP'
scores = {}
prediction = {}

if not os.path.exists(path_data):
    os.makedirs(path_data)

Logger().log('Processing %s' % name)


Logger().reset()
models = []
folds = 10
test_size = 0.1
seed = 13

for b, l, s in Logger().progressbar(it=parameters, end='Finish'):
    # gets name ensemble
    name_model = name + ' %.2g %.2g %.2g' % (b, l, s)
    # make dirs
    _dir = path_data + name_model + '/'
    make_dirs(_dir)
    # generate ensemble
    file_model = _dir + 'model.pkl'
    file_data = _dir + 'data.pkl'

    if not os.path.exists(file_data):

        Logger().log_disable()

        _model = get_ensemble_cip(_name=name_model, _beta=b, _lamb=l, s=s)

        m_scores = []
        m_prediction = []

        if not os.path.exists(file_model):
            # Save ensemble
            models.append(_model)
            _model.save(file_model)
        else:
            # Load model
            _model.load(file_model)

        for fold in range(folds):

            Logger().log('Fold: %d' % (fold + 1))
            file_sets_fold = path_data + 'sets_fold_%d.pkl' % fold
            if not os.path.exists(file_sets_fold):
                # Generate testing and training sets
                input_train, input_test, target_train, target_test = \
                    cross_validation.train_test_split(data_input, data_target, test_size=test_size,
                                                      stratify=data_target, random_state=seed)
                sets_data = Serializable((input_train, input_test, target_train, target_test))
                sets_data.save(file_sets_fold)
            else:
                # Load sets
                Logger().log('Load sets: %s' % file_sets_fold)
                sets_data = Serializable()
                sets_data.load(file_sets_fold)
                input_train, input_test, target_train, target_test = sets_data.get_data()

            file_model_fold = _dir + 'data_fold_%d.pkl' % fold
            score_train, score_test = get_scores(_model, file_model_fold,
                                                 input_train, target_train, input_test, target_test, **args_train)

            m_scores.append((score_train, score_test))

            pred_models = []
            for m in _model.get_models():
                pred_models.append(((m.predict(input_train), m.predict(input_test))))

            m_prediction.append(pred_models)

        data = {'scores': m_scores, 'prediction': m_prediction}
        s_data = Serializable(data)
        s_data.save(file_data)

        Logger().log_enable()
    else:
        Logger().log('Load file: %s' % file_data)
        s_data = Serializable()
        s_data.load(file_data)
        data = s_data.get_data()

    scores[(b, l, s)] = data['scores']
    prediction[(b, l, s)] = data['prediction']


def get_best_score(_scores, _s):
    best_scores = []
    beta = []
    lamb = []
    for key, value in _scores.items():
        if abs(key[2] - _s) < 0.0001:
            best_scores.append(np.max(value, axis=0)[1])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


def get_mean_score(_scores, _s, train=False):
    best_scores = []
    beta = []
    lamb = []
    i = 0 if train else 1
    for key, value in _scores.items():
        if abs(key[2] - _s) < 0.0001:
            d = np.mean(value, axis=0)
            best_scores.append(d[i])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


def get_mean_diversity(_predictions, _s, train=False):
    best_scores = []
    beta = []
    lamb = []
    i = 0 if train else 1
    for key, value in _predictions.items():
        if abs(key[2] - _s) < 0.0001:
            d = np.mean(value, axis=0)
            best_scores.append(d[i])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


def plot_graph(fig, ax, X, Y, Z, xlabel, ylabel, zlabel):

    x = np.linspace(min(X), max(X), 25)
    y = np.linspace(min(Y), max(Y), 25)

    zz = griddata(X, Y, Z, x, y, interp='linear')
    xx, yy = np.meshgrid(x, y)

    x = xx.flatten()
    y = yy.flatten()
    z = zz.flatten()
    z_max = np.max(z)
    z_min = np.min(z)

    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    surf.set_clim([z_min, z_max])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(z_min, z_max)

    fig.colorbar(surf, ticks=np.linspace(z_min, z_max, 5))

def plot_graph2(fig, ax, X, Y, Z, xlabel, ylabel, zlabel):

    x = np.linspace(min(X), max(X), 100)
    y = np.linspace(min(Y), max(Y), 100)

    zz = griddata(X, Y, Z, x, y, interp='linear')
    xx, yy = np.meshgrid(x, y)

    x = xx.flatten()
    y = yy.flatten()
    z = zz.flatten()

    p = ax.pcolor(xx, yy, zz, cmap=cm.jet, vmin=abs(zz).min(), vmax=abs(zz).max())
    cb = fig.colorbar(p)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

fig, _ = plt.subplots()
for i, s1 in enumerate(ss):
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = plt.subplot(3, 2, i + 1)
    X, Y, Z = get_mean_score(scores, s1, False)
    plot_graph2(fig, ax, X, Y, Z, r'$\lambda$', r'$\beta$', r'Accuracy')

plt.show()