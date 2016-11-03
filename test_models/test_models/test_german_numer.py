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
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('germannumer_scale', data_home='../data')

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
n_neurons = int(0.75 * (n_features + n_output)) * n_ensemble_models

n_estimators_rf = int(math.sqrt(n_features))
n_estimators_ensemble_rf = max(n_estimators_rf // n_ensemble_models, 10)
n_neurons_ensemble_per_models = n_neurons // n_ensemble_models

lr = 0.05
reg_l1 = 0.0001
reg_l2 = 0.0001

#############################################################################################################
# Define Parameters training
#############################################################################################################

max_epoch = 300
folds = 1
batch_size = 32
training = True

#############################################################################################################
# Define Models
#############################################################################################################

models = []

# ==========< Ensemble        >==============================================================================
ensemble = EnsembleModel(name='Ensemble')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(mse, name="MSE")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensemble.append_model(net)

ensemble.set_combiner(PluralityVotingCombiner())
models.append(ensemble)

# ==========< Ensemble  Neg Correntropy   >=================================================================
ensembleNCPY = EnsembleModel(name='Ensemble Neg Correntropy')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(mse, name="MSE")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleNCPY.append_model(net)

ensembleNCPY.set_combiner(PluralityVotingCombiner())
ensembleNCPY.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.4)
models.append(ensembleNCPY)

# ==========< Ensemble  MCC - Neg Correntropy   >============================================================
ensembleNCPY_MCC = EnsembleModel(name='Ensemble MCC - Neg Correntropy')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(mcc, name="MCC")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleNCPY_MCC.append_model(net)

ensembleNCPY_MCC.set_combiner(PluralityVotingCombiner())
ensembleNCPY_MCC.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.4)
models.append(ensembleNCPY_MCC)


# ==========< Ensemble  KLG - Neg Correntropy   >============================================================
ensembleNCPY_KLG = EnsembleModel(name='Ensemble KLG - Neg Correntropy')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleNCPY_KLG.append_model(net)

ensembleNCPY_KLG.set_combiner(PluralityVotingCombiner())
ensembleNCPY_KLG.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.5)
models.append(ensembleNCPY_KLG)

# ==========< Ensemble  NCL   >==============================================================================
ensembleNCL = EnsembleModel(name='Ensemble NCL')
ensembleNCL.append_comment("This configuration is the same that paper of"
                           "'Negative Correlation Learning (NCL)' Xin Yao, 1999.")
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(mse, name="MSE")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleNCL.append_model(net)

ensembleNCL.add_cost_ensemble(fun_cost=neg_corr, name="NEG_CORR", lamb_neg_corr=1.0)
ensembleNCL.set_combiner(PluralityVotingCombiner())
models.append(ensembleNCL)

# ==========< Ensemble  NCL L1+L2   >========================================================================
ensembleNCL_L1L2 = EnsembleModel(name='Ensemble NCL L1+L2')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(mse, name="MSE")
    net.append_reg(L1, name='Regularization L1', lamb=reg_l1)
    net.append_reg(L2, name='Regularization L2', lamb=reg_l2)
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleNCL_L1L2.append_model(net)

ensembleNCL_L1L2.add_cost_ensemble(fun_cost=neg_corr, name="NEG_CORR", lamb_neg_corr=0.6)

ensembleNCL_L1L2.set_combiner(PluralityVotingCombiner())
models.append(ensembleNCL_L1L2)

# ==========< Ensemble Kullback Leibler Generalized    >====================================================
ensembleKLG = EnsembleModel(name='Ensemble KLG')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleKLG.append_model(net)

ensembleKLG.set_combiner(PluralityVotingCombiner())
models.append(ensembleKLG)

# ==========< Ensemble MCC    >=============================================================================
ensembleMCC = EnsembleModel(name='Ensemble MCC')
for i in range(n_ensemble_models):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                        activation=fn_activation))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation))
    net.append_cost(mcc, name="MCC")
    net.set_update(sgd, name="SGD", learning_rate=lr)
    ensembleMCC.append_model(net)

ensembleMCC.set_combiner(PluralityVotingCombiner())
models.append(ensembleMCC)

# ==========< MLP 40  MSE  >===================================================================================
net40_MSE = Sequential("MLP %d MSE" % n_neurons, "classifier", classes_labels)
net40_MSE.add_layer(Dense(n_input=n_inputs, n_output=n_neurons,
                          activation=fn_activation))
net40_MSE.add_layer(Dense(n_output=n_output, activation=fn_activation))
net40_MSE.append_cost(mse, name="MSE")
net40_MSE.set_update(sgd, name="SGD", learning_rate=lr)

models.append(net40_MSE)

# ==========< MLP 40 ADAGRAD MSE  >===========================================================================
net40_ADAGRAD_MSE = Sequential("MLP %d ADAGRAD MSE" % n_neurons, "classifier", classes_labels)
net40_ADAGRAD_MSE.add_layer(Dense(n_input=n_inputs, n_output=n_neurons,
                                  activation=fn_activation))
net40_ADAGRAD_MSE.add_layer(Dense(n_output=n_output, activation=fn_activation))
net40_ADAGRAD_MSE.append_cost(mse, name="MSE")
net40_ADAGRAD_MSE.append_reg(L1, name='Regularization L1', lamb=reg_l1)
net40_ADAGRAD_MSE.append_reg(L2, name='Regularization L2', lamb=reg_l2)
net40_ADAGRAD_MSE.set_update(adagrad, name="Adagrad", initial_learning_rate=lr)

models.append(net40_ADAGRAD_MSE)

# ==========< MLP 40 KLG  >==================================================================================
net40_KLG = Sequential("MLP %d KLG" % n_neurons, "classifier", classes_labels)
net40_KLG.add_layer(Dense(n_input=n_inputs, n_output=n_neurons,
                          activation=fn_activation))
net40_KLG.add_layer(Dense(n_output=n_output, activation=fn_activation))
net40_KLG.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
net40_KLG.set_update(sgd, name="SGD", learning_rate=lr)

models.append(net40_KLG)

# ==========< MLP 40 MCC  >==================================================================================
net40_MCC = Sequential("MLP %d MCC" % n_neurons, "classifier", classes_labels)
net40_MCC.add_layer(Dense(n_input=n_inputs, n_output=n_neurons,
                          activation=fn_activation))
net40_MCC.add_layer(Dense(n_output=n_output, activation=fn_activation))
net40_MCC.append_cost(mcc, name="MCC")
net40_MCC.set_update(sgd, name="SGD", learning_rate=lr)

models.append(net40_MCC)

# ==========< MLP 10 ADAGRAD MSE  >===========================================================================
net10_ADAGRAD_MSE = Sequential("MLP %d ADAGRAD MSE" % n_neurons_ensemble_per_models, "classifier", classes_labels)
net10_ADAGRAD_MSE.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                                  activation=fn_activation))
net10_ADAGRAD_MSE.add_layer(Dense(n_output=n_output, activation=fn_activation))
net10_ADAGRAD_MSE.append_cost(mse, name="MSE")
net10_ADAGRAD_MSE.append_reg(L1, name='Regularization L1', lamb=reg_l1)
net10_ADAGRAD_MSE.append_reg(L2, name='Regularization L2', lamb=reg_l2)
net10_ADAGRAD_MSE.set_update(adagrad, name="Adagrad", initial_learning_rate=lr)

models.append(net10_ADAGRAD_MSE)

# ==========< MLP 10 KLG  >===================================================================================
net10_KLG = Sequential("MLP %d KLG" % n_neurons_ensemble_per_models, "classifier", classes_labels)
net10_KLG.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                          activation=fn_activation))
net10_KLG.add_layer(Dense(n_output=n_output, activation=fn_activation))
net10_KLG.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
net10_KLG.set_update(sgd, name="SGD", learning_rate=lr)

models.append(net10_KLG)

# ==========< MLP 10 MCC  >===================================================================================
net10_MCC = Sequential("MLP %d MCC" % n_neurons_ensemble_per_models, "classifier", classes_labels)
net10_MCC.add_layer(Dense(n_input=n_inputs, n_output=n_neurons_ensemble_per_models,
                          activation=fn_activation))
net10_MCC.add_layer(Dense(n_output=n_output, activation=fn_activation))
net10_MCC.append_cost(mcc, name="MCC")
net10_MCC.set_update(sgd, name="SGD", learning_rate=lr)

models.append(net10_MCC)

# ==========< Random Forest  >=================================================================================
rf = RandomForestClassifier(n_estimators=n_estimators_rf)
random_forest = Wrapper(rf, name='Random Forest',
                        input_shape=(n_inputs,), output_shape=(n_output,),
                        type_model='classifier', target_labels=classes_labels)
random_forest.append_comment('Random Forest with %d estimators. This algorithm is implemented for Scikit library.'
                             % n_estimators_rf)

models.append(random_forest)

# ==========< Random Forest Ensemble  >========================================================================
ensembleRandomForest = EnsembleModel(name='Ensemble with Random Forest')
for i in range(n_ensemble_models):
    rft = RandomForestClassifier(n_estimators=n_estimators_ensemble_rf)
    rfw = Wrapper(rft, name='Random Forest %d' % (i + 1),
                  input_shape=(n_inputs,), output_shape=(n_output,),
                  type_model='classifier', target_labels=classes_labels)
    ensembleRandomForest.append_model(rfw)

ensembleRandomForest.set_combiner(PluralityVotingCombiner())
models.append(ensembleRandomForest)

# ==========< SVM  >===========================================================================================
svmc = svm.SVC()
svmw = Wrapper(svmc, name='SVM kernel RBF',
               input_shape=(n_inputs,), output_shape=(n_output,),
               type_model='classifier', target_labels=classes_labels)

models.append(svmw)

# ==========< Ensemble SVM  >==================================================================================
ensembleSVM = EnsembleModel(name='Ensemble with SVM kernel RBF')
for i in range(n_ensemble_models):
    svmt = svm.SVC()
    svmwt = Wrapper(svmt, name='SVM %d' % (i + 1),
                    input_shape=(n_inputs,), output_shape=(n_output,),
                    type_model='classifier', target_labels=classes_labels)
    ensembleSVM.append_model(svmwt)

ensembleSVM.set_combiner(PluralityVotingCombiner())
models.append(ensembleSVM)

models = [ensemble]

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
    # Arguments Training  >==================================================================================
    args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': True,
                  'improvement_threshold': 0.9995, 'update_sets': False}

    # Training Models >======================================================================================
    data_models = test_models(models=models,
                              input_train=input_train, target_train=target_train, input_valid=input_test,
                              target_valid=target_test,
                              classes_labels=classes_labels, name_db=name_db, desc=desc, col_names=col_names,
                              folds=folds, **args_train)
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

plot_scores_classifications(models, input_train, target_train, input_test, target_test, classes_labels)

plt.show()