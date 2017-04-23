import os
import sys

# noinspection PyPep8
import matplotlib.pyplot as plt
# noinspection PyPep8
import numpy as np
# noinspection PyPep8
from sklearn import model_selection

# noinspection PyPep8
from theano import shared, config

sys.path.insert(0, os.path.abspath('../../..'))  # load deepensemble library

# noinspection PyPep8
from deepensemble.utils import load_data, plot_data_training_ensemble
# noinspection PyPep8
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
# noinspection PyPep8
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
# noinspection PyPep8
from deepensemble.utils.utils_models import get_ensembleCIP_model
# noinspection PyPep8
from deepensemble.utils.update_functions import sgd

config.optimizer = 'fast_compile'
# config.exception_verbosity='high'
# config.compute_test_value='warn'

#############################################################################################################
# Load Data
#############################################################################################################
# data_db = load_data_segment(data_home='../../data', normalize=True)
data_db = load_data('australian_scale', data_home='../../data', normalize=False)
# data_db = load_data_iris()
data_input, data_target, classes_labels, name_db, desc, col_names = data_db

# Generate testing and training sets
input_train, input_test, target_train, target_test = \
    model_selection.train_test_split(data_input, data_target, test_size=0.3)

#############################################################################################################
# Define Parameters nets
#############################################################################################################

n_features = data_input.shape[1]
n_classes = len(classes_labels)

n_output = n_classes
n_inputs = n_features

n_neurons_model = int(0.5 * (n_output + n_inputs))

n_ensemble_models = 5
fn_activation1 = ActivationFunctions.sigmoid
fn_activation2 = ActivationFunctions.tanh

y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y))).eval()

#############################################################################################################
# Testing
#############################################################################################################

# ==========< Ensemble  CIP   >===============================================================================
ensembleCIP = get_ensembleCIP_model(name='Ensamble CIP',
                                    n_input=n_features, n_output=n_output,
                                    n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                    classification=True,
                                    is_cip_full=True,
                                    classes_labels=classes_labels,
                                    fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                    dist='ED-CIP',
                                    beta=0, lamb=0, s=s,
                                    bias_layer=True, mse_first_epoch=False, annealing_enable=True,
                                    update=sgd, name_update='SGD',
                                    params_update={'learning_rate': -0.5}
                                    )

ensembleCIP.compile(fast=False)

max_epoch = 500
args_train = {'max_epoch': max_epoch, 'batch_size': 32, 'early_stop': False,
              'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True}

metrics = ensembleCIP.fit(input_train, target_train, **args_train)

plot_data_training_ensemble(ensembleCIP, max_epoch, input_train, input_test, target_train, target_test, metrics)
