import matplotlib
matplotlib.use('Qt4Agg')  # debug

import numpy as np
from sklearn import model_selection

from theano import shared, config

from deepensemble.utils import load_data, plot_data_training_ensemble
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from deepensemble.utils.utils_models import get_ensembleCIP_model
from deepensemble.utils.update_functions import sgd
from deepensemble.utils.score_functions import mutual_information_cs, mutual_information_ed, mutual_information_parzen

config.optimizer = 'fast_compile'
# config.exception_verbosity='high'
# config.compute_test_value='warn'

#############################################################################################################
# Load Data
#############################################################################################################
#data_db = load_data_segment(data_home='../../data', normalize=True)
data_db = load_data('australian_scale', data_home='../../data', normalize=False)
#data_db = load_data_iris()
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

n_ensemble_models = 3
fn_activation1 = ActivationFunctions.sigmoid
fn_activation2 = ActivationFunctions.sigmoid

y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y))).eval()

list_scores = [
    {'fun_score': mutual_information_parzen, 'name': 'Mutual Information'},
    {'fun_score': mutual_information_cs, 'name': 'QMI CS'},
    {'fun_score': mutual_information_ed, 'name': 'QMI ED'}
]

#############################################################################################################
# Testing
#############################################################################################################

# ==========< Ensemble  CIP   >===============================================================================
ensembleCIP = get_ensembleCIP_model(name='Ensamble CIP',
                                    n_input=n_features, n_output=n_output,
                                    n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                    classification=True,
                                    is_cip_full=False,
                                    classes_labels=classes_labels,
                                    fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                    dist='CS',
                                    beta=0.3, lamb=0.3, s=s,
                                    lsp=1., lsm=0.1,
                                    bias_layer=False, mse_first_epoch=False, annealing_enable=True,
                                    update=sgd, name_update='SGD',
                                    params_update={'learning_rate': -0.1},  # maximization
                                    list_scores=list_scores
                                    )

ensembleCIP.compile(fast=False)

max_epoch = 500
args_train = {'max_epoch': max_epoch, 'batch_size': 32, 'early_stop': False,
              'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True,
              'criterion_update_params': 'cost', 'maximization_criterion': True}

metrics = ensembleCIP.fit(input_train, target_train, **args_train)

plot_data_training_ensemble(ensembleCIP, max_epoch, input_train, input_test, target_train, target_test, metrics)
