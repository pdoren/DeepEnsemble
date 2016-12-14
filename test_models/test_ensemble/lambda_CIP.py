import os

import matplotlib.pylab as plt
import numpy as np
from sklearn import cross_validation

from deepensemble.utils import *
from deepensemble.utils.utils_functions import ActivationFunctions
from theano import shared

SEED = 13
plt.style.use('ggplot')

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = \
    load_data('germannumer_scale', data_home='../data')

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
n_neurons = (n_output + n_inputs) // 2

n_neurons_ensemble_per_models = n_neurons // n_ensemble_models

lr = 0.02

#############################################################################################################
# Define Parameters training
#############################################################################################################

max_epoch = 300
folds = 5
batch_size = 40

args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': True,
              'improvement_threshold': 0.9995, 'update_sets': True}


# ==========< Ensemble CIP  >================================================================================
def get_ensemble_cip(_name, _beta, _lamb, s, fast=True):
    ensemble = ensembleCIP_classification(name=_name,
                                          n_feature=n_features, classes_labels=classes_labels,
                                          n_ensemble_models=n_ensemble_models,
                                          n_neurons_ensemble_per_models=n_neurons_ensemble_per_models,
                                          fn_activation1=ActivationFunctions.tanh,
                                          fn_activation2=ActivationFunctions.sigmoid,
                                          beta=_beta, lamb=_lamb, s=s, lr=lr)
    ensemble.compile(fast=fast)

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################

y = get_index_label_classes(translate_target(data_target, classes_labels))
silverman = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()

ss = np.array([silverman * 10**i for i in range(-2, 3)])
ss = [silverman]
beta = np.linspace(0, 1, 8)
lamb = np.linspace(0, 1, 8)

bb, ll, sss = np.meshgrid(beta, lamb, ss)
parameters = list(zip(bb.flatten(), ll.flatten(), sss.flatten()))



path_data = 'test_params_cip/'

name = 'Ensemble CIP'
scores = {}

if not os.path.exists(path_data):
    os.makedirs(path_data)
Logger().log('Processing %s' % name)
file_scores = path_data + 'scores_%s_%s.pkl' % (name, name_db)

if not os.path.exists(file_scores):
    Logger().reset()
    for b, l, s in Logger().progressbar(it=parameters, end='Finish'):
        Logger().log_disable()
        model = get_ensemble_cip(_name=name, _beta=b, _lamb=l, s=s)

        _, best_score, list_score = test_model(cls=model,
                                             input_train=input_train, target_train=target_train,
                                             input_test=input_test, target_test=target_test,
                                             folds=folds, **args_train)

        scores[(b, l, s)] = {'best_score': best_score, 'list_score': list_score}
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

z = griddata(X, Y, Z, x, y, interp='linear')
x, y = np.meshgrid(x, y)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

x = x.flatten()
y = y.flatten()
z = z.flatten()
ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$\beta$')
ax.set_zlabel(r'Accuracy')

plt.show()