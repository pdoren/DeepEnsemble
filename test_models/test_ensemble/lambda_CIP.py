import os

import matplotlib.pylab as plt
import numpy as np

from deepensemble.utils import *
from deepensemble.utils.utils_test import make_dirs
from deepensemble.utils.utils_functions import ActivationFunctions
from theano import shared

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

n_ensemble_models = 4
n_neurons = (n_output + n_inputs) // 2

n_neurons_ensemble_per_models = n_neurons // n_ensemble_models

lr = 0.02

#############################################################################################################
# Define Parameters training
#############################################################################################################

# 10-Cross Validation, 300 epoch and 40 size mini-batch
args_train = {'max_epoch': 300, 'batch_size': 40, 'early_stop': True, 'test_size': 0.1,
              'improvement_threshold': 0.9995, 'update_sets': True}


# ==========< Ensemble CIP  >================================================================================
def get_ensemble_cip(_name, _beta, _lamb, s):
    ensemble = ensembleCIP_classification(name=_name,
                                          n_feature=n_features, classes_labels=classes_labels,
                                          n_ensemble_models=n_ensemble_models,
                                          n_neurons_ensemble_per_models=n_neurons_ensemble_per_models,
                                          fn_activation1=ActivationFunctions.tanh,
                                          fn_activation2=ActivationFunctions.sigmoid,
                                          beta=_beta, lamb=_lamb, s=s, lr=lr)

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################

y = get_index_label_classes(translate_target(data_target, classes_labels))
silverman = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()

ss = silverman * np.array([0.01, 0.1, 1, 5, 20 ])
beta = np.linspace(0, 1, 6)
lamb = np.linspace(0, 1, 6)

bb, ll, sss = np.meshgrid(beta, lamb, ss)
parameters = list(zip(bb.flatten(), ll.flatten(), sss.flatten()))

path_data = 'test_params_cip/%s/' % name_db

name = 'Ensemble CIP'
scores = {}

if not os.path.exists(path_data):
    os.makedirs(path_data)

Logger().log('Processing %s' % name)
file_scores = path_data + 'scores_%s.pkl' % name

if not os.path.exists(file_scores):
    Logger().reset()
    models = []
    for b, l, s in Logger().progressbar(it=parameters, end='Finish'):
        Logger().log_disable()
        # gets name ensemble
        name_model = name + ' %.2g %.2g %.2g' % (b, l, s)
        # make dirs
        _dir = path_data + name_model + '/'
        make_dirs(_dir)
        # generate ensemble
        file_model = _dir + 'model.pkl'
        _model = get_ensemble_cip(_name=name_model, _beta=b, _lamb=l, s=s)

        if not os.path.exists(file_model):
            # Compile and save ensemble
            models.append(_model)
            _model.compile(fast=True)
            _model.save(file_model)
        else:
            # Load model
            Logger().log('Load Model: %s' % file_model)
            _model.load(file_model)

    scores = cross_validation_score(models, data_input, data_target,
                                        folds=10, path_db=path_data, **args_train)

    scores[(b, l, s)] = {'best_score': np.max(scores, axis=0)[1], 'list_score': scores}
    Logger().log_enable()

    s_data = Serializable(scores)
    s_data.save(file_scores)
else:
    Logger().log('Load file: %s' % file_scores)
    s_data = Serializable()
    s_data.load(file_scores)
    scores = s_data.get_data()

    #scores = scores.item()
    #print(type(scores))
    #s1_data = Serializable(scores)
    #s1_data.save(file_scores)


def get_best_score(_scores, _s):
    best_scores = []
    beta = []
    lamb = []
    for key, value in _scores.items():
        if abs(key[2] - _s) < 0.0001:
            best_scores.append(value['best_score'])
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
            d = np.mean(value['list_score'], axis=0)
            best_scores.append(d[i])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


s1 = silverman

X, Y, Z = get_mean_score(scores, s1, False)

x = np.linspace(min(X), max(X), 25)
y = np.linspace(min(Y), max(Y), 25)

from matplotlib.mlab import griddata

zz = griddata(X, Y, Z, x, y, interp='linear')
xx, yy = np.meshgrid(x, y)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

x = xx.flatten()
y = yy.flatten()
z = zz.flatten()
z_max = np.max(z)
z_min = np.min(z)

surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
surf.set_clim([z_min, z_max])

ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$\beta$')
ax.set_zlabel(r'Accuracy')
ax.set_zlim(z_min, z_max)

fig.colorbar(surf, ticks=np.linspace(z_min, z_max, 5))

plt.show()