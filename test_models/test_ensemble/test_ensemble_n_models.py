import os

import matplotlib.pylab as plt
import numpy as np
from sklearn import cross_validation

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

n_neurons_ensemble_per_models = n_neurons // n_ensemble_models

lr = 0.01
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
def get_ensemble_ncl(_name, _n_models, fast=True):
    ensemble = ensembleNCL_classification(name=_name,
                                          input_train=input_train,
                                          classes_labels=classes_labels,
                                          n_ensemble_models=_n_models,
                                          n_neurons_models=n_neurons_ensemble_per_models,
                                          fn_activation1=ActivationFunctions.tanh,
                                          fn_activation2=ActivationFunctions.sigmoid,
                                          lamb=0.6, lr=lr)
    ensemble.compile(fast=fast)

    return ensemble


def get_ensemble_cip(_name, _n_models, fast=True):
    ensemble = ensembleCIP_classification(name=_name,
                                          n_feature=input_train, classes_labels=classes_labels,
                                          n_ensemble_models=_n_models,
                                          n_neurons_models=n_neurons_ensemble_per_models,
                                          fn_activation1=ActivationFunctions.tanh,
                                          fn_activation2=ActivationFunctions.sigmoid,
                                          beta=0.3, lr=5 * lr)
    ensemble.compile(fast=fast)

    return ensemble


#############################################################################################################
#  TEST
#############################################################################################################
parameters = [n for n in range(1, 20, 3)]

list_ensemble = [(get_ensemble_ncl, 'Ensemble NCL'), (get_ensemble_cip, 'Ensemble CIP')]
path_data = 'test_ensemble_n_models/'

for get_ensemble, name in list_ensemble:

    data = {}
    scores = []

    if not os.path.exists(path_data):
        os.makedirs(path_data)
    Logger().log('Processing %s' % name)
    file_data = path_data + '%s_%s.pkl' % (name, name_db)

    if not os.path.exists(file_data):
        Logger().reset()
        for _p in Logger().progressbar(it=parameters, end='Finish'):
            Logger().log_disable()
            model = get_ensemble(_name=name, _n_models=_p)

            metrics, best_score, list_score = test_model(cls=model,
                                                         input_train=input_train, target_train=target_train,
                                                         input_test=input_test, target_test=target_test,
                                                         folds=folds, **args_train)

            scores.append(best_score)
            data[_p] = {'model': model, 'metrics': metrics, 'list_score': list_score}
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

    f, ax = plt.subplots()
    plt.hold(True)
    list_dp = []

    for i in range(folds):
        y = list([data[l]['list_score'][i] for l in parameters])
        x = list(np.array(parameters) / n_features)
        dp = DataPlot(name=name, _type='score')
        dp.set_data(x, y)
        list_dp.append(dp)

    plot_data(ax, [(list_dp, 'score')],
              x_max=max(parameters) / n_features, x_min=min(parameters) / n_features,
              title='%s Accuracy' % name)

    plt.xlabel('$n^o models / n^o features$')
    plt.ylabel('score')
    plt.tight_layout()

plt.show()
