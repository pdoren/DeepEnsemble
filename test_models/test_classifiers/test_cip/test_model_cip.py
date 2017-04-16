import os
import sys

sys.path.insert(0, os.path.abspath('../../..'))  # load deepensemble library

import math

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import  model_selection
from sklearn.metrics import accuracy_score
from theano import shared, config
import theano.tensor as T
from collections import OrderedDict

from deepensemble.utils import load_data, plot_pdf, load_data_segment, load_data_iris
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from deepensemble.utils.utils_models import get_ensembleCIP_model
from deepensemble.utils.update_functions import sgd_cip

config.optimizer='fast_compile'
#config.exception_verbosity='high'
#config.compute_test_value='warn'

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

n_ensemble_models = 1
fn_activation1 = ActivationFunctions.sigmoid
fn_activation2 = ActivationFunctions.tanh

y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()


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
                                    dist='ED-CIP',
                                    beta=0, lamb=0, s=s,
                                    bias_layer=False, mse_first_epoch=False, annealing_enable=False,
                                    update=sgd_cip, name_update='SGD CIP',
                                    params_update={'learning_rate': 0.01}
                                    )

ensembleCIP.compile(fast=False)

max_epoch = 500
args_train = {'max_epoch': max_epoch, 'batch_size': 20, 'early_stop': False,
              'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True}

metrics = ensembleCIP.fit(input_train, target_train, **args_train)

e_train = ensembleCIP.error(input_train, ensembleCIP.translate_target(target_train)).eval()
e_test = ensembleCIP.error(input_test, ensembleCIP.translate_target(target_test)).eval()

plt.style.use('ggplot')
f = plt.figure()

ax = plt.subplot(2, 1, 1)
for i in range(n_output):
    plot_pdf(ax, e_test[:, i], label='Test output %d' % (i + 1), x_min=-2, x_max=2, n_points=1000)
plt.legend()

ax = plt.subplot(2, 1, 2)
for i in range(n_output):
    plot_pdf(ax, e_train[:, i], label='Train output %d' % (i + 1), x_min=-2, x_max=2, n_points=1000)
plt.legend()

# noinspection PyRedeclaration
f = plt.figure()
msg_train = ''
msg_test = ''
row = math.ceil(n_ensemble_models / 2.0)
col = 2
for i, model in enumerate(ensembleCIP.get_models()):
    e_train = model.error(input_train, model.translate_target(target_train)).eval()
    e_test = model.error(input_test, model.translate_target(target_test)).eval()

    ax = plt.subplot(row, col, i + 1)
    for j in range(n_output):
        plot_pdf(ax, e_test[:, j], label='Test output %d' % (j + 1), x_min=-2, x_max=2, n_points=1000)
    plt.legend()
    plt.title('Model %s' % model.get_name())

    pred_test = model.predict(input_test)
    pred_train = model.predict(input_train)

    msg_test += 'Accuracy model %s test: %.4g\n' %\
                (model.get_name(), accuracy_score(pred_test, target_test))
    msg_train += 'Accuracy model %s train: %.4g\n' %\
                 (model.get_name(), accuracy_score(pred_train, target_train))

print(msg_test)
print('Accuracy Ensemble test: %.4g' % (accuracy_score(ensembleCIP.predict(input_test), target_test)))
print('--' * 10)
print(msg_train)
print('Accuracy Ensemble train: %.4g' % (accuracy_score(ensembleCIP.predict(input_train), target_train)))

plt.tight_layout()

metrics.plot_cost(max_epoch=max_epoch, title='Costo CIP')
metrics.plot_costs(max_epoch=max_epoch, title='Costo CIP')
metrics.plot_scores(max_epoch=max_epoch, title='Desempeño CIP')

om_train = ensembleCIP.output(input_train).eval()
om_test = ensembleCIP.output(input_test).eval()

# noinspection PyRedeclaration
#f = plt.figure()
#ax = plt.subplot(2, 1, 1)
#ax.plot(om_train[:, 0] - om_train[:, 1], '.', label='Train')
#plt.legend()

#ax = plt.subplot(2, 1, 2)
#ax.plot(om_test[:, 0] - om_test[:, 1], '.', label='Test')
#plt.legend()

plt.show()
