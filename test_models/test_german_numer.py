from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, cross_validation

import matplotlib.pylab as plt

from deepensemble.combiner import *
from deepensemble.layers import *
from deepensemble.models import *
from deepensemble.utils import *
from deepensemble.utils.utils_functions import ActivationFunctions

SEED = 13
plt.style.use('ggplot')

training = True

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('germannumer_scale')

input_train, input_test, target_train, target_test = \
            cross_validation.train_test_split(data_input, data_target, test_size=0.3, stratify=data_target,
                                              random_state=SEED)

#############################################################################################################
# Define Models
#############################################################################################################

models = []

# ==========< Ensemble        >==============================================================================
ensemble = EnsembleModel(name='Ensemble')
ensemble.set_info('Ensemble with 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                  'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                  'the activation function is sigmoid for each neurons. The cost function is MSE.\n'
                  ' The combiner output model ensemble is Max Voting or Plurality Voting.')
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mse, name="MSE")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensemble.append_model(net)

ensemble.set_combiner(PluralityVotingCombiner())
models.append(ensemble)

# ==========< Ensemble  Neg Correntropy   >=================================================================
ensembleNCPY = EnsembleModel(name='Ensemble Neg Correntropy')
ensembleNCPY.set_info('Ensemble training with Negative Correntropy and MSE.\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons. The cost function is MSE.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting.')
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mse, name="MSE")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleNCPY.append_model(net)

ensembleNCPY.set_combiner(PluralityVotingCombiner())
ensembleNCPY.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.4)

# ==========< Ensemble  MCC - Neg Correntropy   >============================================================
ensembleNCPY_MCC = EnsembleModel(name='Ensemble MCC - Neg Correntropy')
ensembleNCPY_MCC.set_info('Ensemble training with Negative Correntropy and MCC.\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons. The cost function is MSE.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting.')
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mcc, name="MCC")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleNCPY_MCC.append_model(net)

ensembleNCPY_MCC.set_combiner(PluralityVotingCombiner())
ensembleNCPY_MCC.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.4)


# ==========< Ensemble  KLG - Neg Correntropy   >============================================================
ensembleNCPY_KLG = EnsembleModel(name='Ensemble KLG - Neg Correntropy')
ensembleNCPY_KLG.set_info('Ensemble training with Negative Correntropy and KLG.\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons. The cost function is MSE.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting.')
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleNCPY_KLG.append_model(net)

ensembleNCPY_KLG.set_combiner(PluralityVotingCombiner())
ensembleNCPY_KLG.add_cost_ensemble(fun_cost=neg_correntropy, name="NEG_CORRPY", lamb_corr=0.5)

# ==========< Ensemble  NCL   >==============================================================================
ensembleNCL = EnsembleModel(name='Ensemble NCL')
ensembleNCL.set_info('Ensemble training with Negative Correlation Learning (NCL).\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons. The cost function is MSE.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting.\n'
                     " This configuration is the same that paper of"
                     " 'Negative Correlation Learning (NCL)' Xin Yao, 1999.")
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mse, name="MSE")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleNCL.append_model(net)

ensembleNCL.add_cost_ensemble(fun_cost=neg_corr, name="NEG_CORR", lamb_neg_corr=0.6)
# The lamb_neg_corr changed from 1.0 to 0.6 for improve score classification

ensembleNCL.set_combiner(PluralityVotingCombiner())
models.append(ensembleNCL)

# ==========< Ensemble  NCL L1+L2   >========================================================================
ensembleNCL_L1L2 = EnsembleModel(name='Ensemble NCL L1+L2')
ensembleNCL_L1L2.set_info('Ensemble training with Negative Correlation Learning (NCL).\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons. The cost function is MSE.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting.\n'
                     " This cost function include MSE and regularization L1 and L2.")
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mse, name="MSE")
    net.append_reg(L1, name='Regularization L1', lamb=0.005)
    net.append_reg(L2, name='Regularization L2', lamb=0.001)
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleNCL_L1L2.append_model(net)

ensembleNCL_L1L2.add_cost_ensemble(fun_cost=neg_corr, name="NEG_CORR", lamb_neg_corr=0.6)

ensembleNCL_L1L2.set_combiner(PluralityVotingCombiner())
models.append(ensembleNCL_L1L2)

# ==========< Ensemble Kullback Leibler Generalized    >====================================================
ensembleKLG = EnsembleModel(name='Ensemble KLG')
ensembleKLG.set_info('Ensemble training with Kullback Leibler Generalized.\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting. and'
                     ' The cost function is Kullback Leibler Generalized.')
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleKLG.append_model(net)

ensembleKLG.set_combiner(PluralityVotingCombiner())
models.append(ensembleKLG)

# ==========< Ensemble MCC    >=============================================================================
ensembleMCC = EnsembleModel(name='Ensemble MCC')
ensembleMCC.set_info('Ensemble training with MCC (Minimum Cross Correntropy).\n'
                     'This ensemble has 4 neural networks type MLP, training algorithm is SGD (lr=0.1).\n'
                     'MLPs: 10 neurons in hidden layers and 2 neurons in output (one hot encoding),\n'
                     ' the activation function is sigmoid for each neurons.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting'
                     ' and the cost function is MCC.')
for i in range(4):
    net = Sequential("net%d" % (i + 1), "classifier", classes_labels)
    net.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mcc, name="MCC")
    net.set_update(sgd, name="SGD", learning_rate=0.1)
    ensembleMCC.append_model(net)

ensembleMCC.set_combiner(PluralityVotingCombiner())
models.append(ensembleMCC)

# ==========< MLP 40  MSE  >===================================================================================
net40_MSE = Sequential("MLP 40 MSE", "classifier", classes_labels)
net40_MSE.set_info('Neural Network type MLP 40 neurons in hidden layers and'
                   ' 2 neurons in output (one hot encoding).\n'
                   ' The training is with SGD (lr=0.1), the activation function is sigmoid for each neurons.\n'
                   ' The cost function is MSE.')
net40_MSE.add_layer(Dense(n_input=data_input.shape[1], n_output=40,
                                  activation=ActivationFunctions.sigmoid))
net40_MSE.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net40_MSE.append_cost(mse, name="MSE")
net40_MSE.set_update(sgd, name="SGD", learning_rate=0.1)

models.append(net40_MSE)

# ==========< MLP 40 ADAGRAD MSE  >===========================================================================
net40_ADAGRAD_MSE = Sequential("MLP 40 ADAGRAD MSE", "classifier", classes_labels)
net40_ADAGRAD_MSE.set_info('Neural Network type MLP 40 neurons in hidden layers and'
                           ' 2 neurons in output (one hot encoding).\n'
                           ' The training is with ADAGRAD, the activation function is sigmoid for each neurons.\n'
                           ' The cost function is MSE with regularization L1 and L2 (lamb: L1: 0.005, L2: 0.001 ).')
net40_ADAGRAD_MSE.add_layer(Dense(n_input=data_input.shape[1], n_output=40,
                                  activation=ActivationFunctions.sigmoid))
net40_ADAGRAD_MSE.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net40_ADAGRAD_MSE.append_cost(mse, name="MSE")
net40_ADAGRAD_MSE.append_reg(L1, name='Regularization L1', lamb=0.005)
net40_ADAGRAD_MSE.append_reg(L2, name='Regularization L2', lamb=0.001)
net40_ADAGRAD_MSE.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)

models.append(net40_ADAGRAD_MSE)

# ==========< MLP 40 KLG  >==================================================================================
net40_KLG = Sequential("MLP 40 KLG", "classifier", classes_labels)
net40_KLG.set_info('Neural Network type MLP 40 neurons in hidden layers and'
                   ' 2 neurons in output (one hot encoding).\n'
                   ' The training is with SGD (lr=0.1), the activation function is sigmoid for each neurons.\n'
                   ' The cost function is Kullback Leibler Generalized.')
net40_KLG.add_layer(Dense(n_input=data_input.shape[1], n_output=40,
                          activation=ActivationFunctions.sigmoid))
net40_KLG.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net40_KLG.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
net40_KLG.set_update(sgd, name="SGD", learning_rate=0.1)

models.append(net40_KLG)

# ==========< MLP 40 MCC  >==================================================================================
net40_MCC = Sequential("MLP 40 MCC", "classifier", classes_labels)
net40_MCC.set_info('Neural Network type MLP 40 neurons in hidden layers and'
                   ' 2 neurons in output (one hot encoding).\n'
                   ' The training is with SGD (lr=0.1), the activation function is sigmoid for each neurons.\n'
                   ' The cost function is MCC.')
net40_MCC.add_layer(Dense(n_input=data_input.shape[1], n_output=40,
                          activation=ActivationFunctions.sigmoid))
net40_MCC.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net40_MCC.append_cost(mcc, name="MCC")
net40_MCC.set_update(sgd, name="SGD", learning_rate=0.1)

models.append(net40_MCC)

# ==========< MLP 10 ADAGRAD MSE  >===========================================================================
net10_ADAGRAD_MSE = Sequential("MLP 10 ADAGRAD MSE", "classifier", classes_labels)
net10_ADAGRAD_MSE.set_info('Neural Network type MLP 10 neurons in hidden layers and'
                           ' 2 neurons in output (one hot encoding).\n'
                           ' The training is with ADAGRAD, the activation function is sigmoid for each neurons.\n'
                           ' The cost function is MSE with regularization L1 and L2.')
net10_ADAGRAD_MSE.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                                  activation=ActivationFunctions.sigmoid))
net10_ADAGRAD_MSE.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net10_ADAGRAD_MSE.append_cost(mse, name="MSE")
net10_ADAGRAD_MSE.append_reg(L1, name='Regularization L1', lamb=0.005)
net10_ADAGRAD_MSE.append_reg(L2, name='Regularization L2', lamb=0.001)
net10_ADAGRAD_MSE.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)

models.append(net10_ADAGRAD_MSE)

# ==========< MLP 10 KLG  >===================================================================================
net10_KLG = Sequential("MLP 10 KLG", "classifier", classes_labels)
net10_KLG.set_info('Neural Network type MLP 10 neurons in hidden layers and'
                   ' 2 neurons in output (one hot encoding).\n'
                   ' The training is with SGD (lr=0.1), the activation function is sigmoid for each neurons.\n'
                   ' The cost function is Kullback Leibler Generalized.')
net10_KLG.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                          activation=ActivationFunctions.sigmoid))
net10_KLG.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net10_KLG.append_cost(kullback_leibler_generalized, name="Kullback Leibler Generalized")
net10_KLG.set_update(sgd, name="SGD", learning_rate=0.1)

models.append(net10_KLG)

# ==========< MLP 10 MCC  >===================================================================================
net10_MCC = Sequential("MLP 10 MCC", "classifier", classes_labels)
net10_MCC.set_info('Neural Network type MLP 10 neurons in hidden layers and'
                   ' 2 neurons in output (one hot encoding).\n'
                   ' The training is with SGD (lr=0.1), the activation function is sigmoid for each neurons.\n'
                   ' The cost function is MCC.')
net10_MCC.add_layer(Dense(n_input=data_input.shape[1], n_output=10,
                          activation=ActivationFunctions.sigmoid))
net10_MCC.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
net10_MCC.append_cost(mcc, name="MCC")
net10_MCC.set_update(sgd, name="SGD", learning_rate=0.1)

models.append(net10_MCC)

# ==========< Random Forest  >=================================================================================
rf = RandomForestClassifier(n_estimators=40)
random_forest = Wrapper(rf, name='Random Forest',
                        input_shape=(data_input.shape[1],), output_shape=(2,),
                        type_model='classifier', target_labels=classes_labels)
random_forest.set_info('Random Forest with 40 estimators. This algorithm is implemented for Scikit library.')

models.append(random_forest)

# ==========< Random Forest Ensemble  >========================================================================
ensembleRandomForest = EnsembleModel(name='Ensemble with Random Forest')
ensembleRandomForest.set_info('Ensemble with 4 Random Forest Classifiers.\n'
                              ' The combiner output model ensemble is Max Voting or Plurality Voting.\n'
                              ' Random Forest has 10 estimators. This algorithm is implemented for Scikit library.')
for i in range(4):
    rft = RandomForestClassifier(n_estimators=10)
    rfw = Wrapper(rft, name='Random Forest %d' % (i + 1),
                  input_shape=(data_input.shape[1],), output_shape=(2,),
                  type_model='classifier', target_labels=classes_labels)
    ensembleRandomForest.append_model(rfw)

ensembleRandomForest.set_combiner(PluralityVotingCombiner())
models.append(ensembleRandomForest)

# ==========< SVM  >===========================================================================================
svmc = svm.SVC()
svmw = Wrapper(svmc, name='SVM kernel RBF',
               input_shape=(data_input.shape[1],), output_shape=(2,),
               type_model='classifier', target_labels=classes_labels)
svmw.set_info('Super Vector Machine with kernel RBF. This algorithm is implemented for Scikit library.')

models.append(svmw)

# ==========< Ensemble SVM  >==================================================================================
ensembleSVM = EnsembleModel(name='Ensemble with SVM kernel RBF')
ensembleSVM.set_info('Ensemble with 4 SVM (RBF) Classifiers.\n'
                     ' The combiner output model ensemble is Max Voting or Plurality Voting.\n'
                     ' Super Vector Machine with kernel RBF. This algorithm is implemented for Scikit library.')
for i in range(4):
    svmt = svm.SVC()
    svmwt = Wrapper(svmt, name='SVM %d' % (i + 1),
                    input_shape=(data_input.shape[1],), output_shape=(2,),
                    type_model='classifier', target_labels=classes_labels)
    ensembleSVM.append_model(svmwt)

ensembleSVM.set_combiner(PluralityVotingCombiner())
models.append(ensembleSVM)

# ============================================================================================================
# Compile models and define extra score functions  >==========================================================

models=[ensembleNCPY_MCC, net10_ADAGRAD_MSE]

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
    args_train = {'max_epoch': 300, 'batch_size': 32, 'early_stop': False,
                  'improvement_threshold': 0.9995, 'update_sets': False}

    # Training Models >======================================================================================
    data_models = test_models(models=models,
                              input_train=input_train, target_train=target_train, input_valid=input_test,
                              target_valid=target_test,
                              classes_labels=classes_labels, name_db=name_db, desc=desc, col_names=col_names,
                              folds=5, **args_train)
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