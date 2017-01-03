import os

import numpy as np
from theano import shared, config
import matplotlib.pyplot as plt

from deepensemble.utils import load_data, Serializable, plot_pdf
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from test_models.test_classifiers.test_classifiers import test_classifiers, show_data_classification
from deepensemble.utils.cost_functions import kullback_leibler_generalized
from deepensemble.utils.update_functions import rmsprop, adadelta, adagrad, adam
from deepensemble.utils.utils_models import get_ensembleCIP_model
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale',
                                                                              data_home='../../data', normalize=False)

# Generate testing and training sets
input_train, input_test, target_train, target_test = \
    cross_validation.train_test_split(data_input, data_target, test_size=0.3)

#############################################################################################################
# Define Parameters nets
#############################################################################################################

n_features = data_input.shape[1]
n_classes = len(classes_labels)

n_output = n_classes
n_inputs = n_features

n_neurons_model = int(0.5 * (n_output + n_inputs))

n_ensemble_models = 4
fn_activation1 = ActivationFunctions.sigmoid
fn_activation2 = ActivationFunctions.sigmoid

y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()
S = s

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
                                 dist='CIP',
                                 # cost=kullback_leibler_generalized, name_cost="KLG",
                                 beta=0.9, lamb=0, s=S, bias_layer=True,
                                 params_update={'learning_rate': 0.01},
                                 # update=adagrad, name_update='ADAGRAD', params_update={'learning_rate': 0.01}
            )

ensembleCIP.compile(fast=False)

max_epoch=500
args_train = {'max_epoch': max_epoch, 'batch_size': 40, 'early_stop': False,
                  'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True}

metrics = ensembleCIP.fit(input_train, target_train, **args_train)

e_train = ensembleCIP.error(input_train, ensembleCIP.translate_target(target_train)).eval()
e_test = ensembleCIP.error(input_test, ensembleCIP.translate_target(target_test)).eval()

plt.style.use('ggplot')
f = plt.figure()

ax = plt.subplot(2, 1, 1)
plot_pdf(ax, e_test[:, 0], label='Test Class 1', x_min=-2, x_max=2, n_points=1000)
plot_pdf(ax, e_test[:, 1], label='Test Class 2', x_min=-2, x_max=2, n_points=1000)
plt.legend()

ax = plt.subplot(2, 1, 2)
plot_pdf(ax, e_train[:, 0], label='Train Class 1', x_min=-2, x_max=2, n_points=1000)
plot_pdf(ax, e_train[:, 1], label='Train Class 2', x_min=-2, x_max=2, n_points=1000)
plt.legend()

f = plt.figure()

for i, model in enumerate(ensembleCIP.get_models()):
    e_train = model.error(input_train, model.translate_target(target_train)).eval()
    e_test = model.error(input_test, model.translate_target(target_test)).eval()

    ax = plt.subplot(2, 2, i + 1)
    plot_pdf(ax, e_test[:, 0], label='Test Class 1', x_min=-2, x_max=2, n_points=1000)
    plot_pdf(ax, e_test[:, 1], label='Test Class 2', x_min=-2, x_max=2, n_points=1000)
    plt.legend()
    plt.title('Model %d' % (i + 1))

    print('Accuracy model %d: %.4g' % (i + 1, accuracy_score(model.predict(input_test), target_test)))

print('Accuracy Ensemble: %.4g' % (accuracy_score(ensembleCIP.predict(input_test), target_test)))

plt.tight_layout()

metrics.plot_cost(max_epoch=max_epoch, title='Costo CIP')
metrics.plot_costs(max_epoch=max_epoch, title='Costo CIP')
metrics.plot_scores(max_epoch=max_epoch, title='Desempeño CIP')


om_train = ensembleCIP.output(input_train).eval()
om_test = ensembleCIP.output(input_test).eval()

f = plt.figure()
ax = plt.subplot(2, 1, 1)
ax.plot(om_train[:, 0] - om_train[:, 1], '.', label='Train')
plt.legend()

ax = plt.subplot(2, 1, 2)
ax.plot(om_test[:, 0] - om_test[:, 1], '.', label='Test')
plt.legend()

plt.show()