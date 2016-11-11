from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, cross_validation

import math

import matplotlib.pylab as plt

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
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale')

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

fn_activation = ActivationFunctions.tanh

n_ensemble_models = 4
n_neurons = int(0.5 * (n_features + n_output)) * 2

n_estimators_rf = int(math.sqrt(n_features))
n_estimators_ensemble_rf = max(n_estimators_rf // n_ensemble_models, 10)
n_neurons_ensemble_per_models = n_neurons // n_ensemble_models

lr = 0.0005
reg_l1 = 0.0001
reg_l2 = 0.0001

#############################################################################################################
# Define Parameters training
#############################################################################################################

max_epoch = 500
folds = 1
batch_size = 50
training = True

args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': False,
              'improvement_threshold': 0.9995, 'update_sets': True}

#############################################################################################################
# Define Models
#############################################################################################################

models = []

# ==========< Ensemble        >==============================================================================
ensemble = EnsembleModel(name='Ensemble')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    # net.add_layer(NoiseLayer(input_shape=(None, n_inputs), avg=0.1, std=0.2))
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models, activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=ActivationFunctions.softmax))
    # net.append_cost(itakura_saito, name="Itakura Saito")
    net.append_cost(kullback_leibler_generalized, name="KLG")
    # net.append_cost(kullback_leibler, name="KL")
    # net.append_cost(mse, name="MSE")
    # net.append_cost(mcc, name="MCC")

    #net.compile()
    #net.fit(input_train, target_train, max_epoch=max_epoch, batch_size=32, early_stop=False)

    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensemble.append_model(net)

# ensemble.set_type_training('bagging')
ensemble.set_combiner(GeometricVotingCombiner())
# ensemble.set_combiner(PluralityVotingCombiner())
ensemble.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb=5.9)
# ensemble.add_cost_ensemble(fun_cost=neg_corr, name="NEG_CORR", lamb=0.7)

models.append(ensemble)

# ============================================================================================================
# Compile models and define extra score functions  >==========================================================

if training:  # compile only if training models
    Logger().reset()
    for model in models:
        # Extra Scores
        if isinstance(model, EnsembleModel):
            model.append_score(score_ensemble_ambiguity, 'Ambiguity')

        # Compile
        model.compile(fast=False)

    Logger().save_buffer('info_%s_compile.txt' % name_db)

#############################################################################################################
#
#  TRAINING ALL MODELS
#
#############################################################################################################

if training:
    # Training Models >======================================================================================
    data_models = test_models(models=models,
                              input_train=input_train, target_train=target_train, input_valid=input_test,
                              target_valid=target_test,
                              classes_labels=classes_labels, name_db=name_db, desc=desc, col_names=col_names,
                              folds=folds, save_file=False,  **args_train)
else:
    # Load Data
    dir = name_db + '/'
    for model in models:
        name = model.get_name()
        model.load(dir + '%s/%s_classifier.pkl' % (name, name))


#############################################################################################################
#
#  PLOT DATA CLASSIFIERS
#
#############################################################################################################

plt.close('all')

# plot_scores_classifications(models, input_train, target_train, input_test, target_test, classes_labels)

model = models[0]
data_models[0]['metrics'].plot_cost(max_epoch)
data_models[0]['metrics'].plot_costs(max_epoch)
data_models[0]['metrics'].plot_scores(max_epoch)

e1 = model.error(input_test, model.translate_target(target_test), prob=True).eval().flatten()
e2 = model.error(input_train, model.translate_target(target_train), prob=True).eval().flatten()

_, ax = plt.subplots()

plot_pdf(ax, e1, 'Test pdf')
plot_pdf(ax, e2, 'Train pdf')

plt.xlabel('Error')
plt.ylabel('PDF of the error')
plt.title("Comparison between error probability density functions")
plt.legend()

_, ax = plt.subplots()
for _model in model.get_models():
    e = _model.error(input_test, model.output(input_test), prob=True).eval().flatten()
    plot_pdf(ax, e, 'Model ' + _model.get_name())

plt.xlabel('Error')
plt.ylabel('PDF of the error')
plt.title("Comparison between error probability density functions")
plt.legend()

plt.show()