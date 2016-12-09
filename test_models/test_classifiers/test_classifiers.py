import matplotlib.pylab as plt

from deepensemble.models import EnsembleModel
from deepensemble.utils.cost_functions import mse
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils.utils_models import *
from deepensemble.utils import Logger, score_ensemble_ambiguity, test_models, plot_scores_classifications

def test_classifiers(name_db, input_train, target_train, input_test, target_test, classes_labels,
                     only_cip=False,
                     lamb_ncl=0.6, beta_cip=0.6, lamb_cip=0.2,
                     fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                     folds=5, lr=0.01, training=True, max_epoch=300, batch_size=40):

    args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': False,
                  'improvement_threshold': 0.9995, 'update_sets': True}

    #############################################################################################################
    # Define Parameters nets
    #############################################################################################################

    n_features = input_train.shape[1]
    n_classes = len(classes_labels)

    n_output = n_classes
    n_inputs = n_features

    n_ensemble_models = 4
    n_neurons = (n_output + n_inputs) // 2

    n_neurons_ensemble_per_models = n_neurons


    #############################################################################################################
    # Define Models
    #############################################################################################################

    models = []

    # ==========< Ensemble        >==============================================================================
    ensemble = ensemble_classification(name='Ensamble',
                                       input_train=input_train,
                                       classes_labels=classes_labels,
                                       n_ensemble_models=n_ensemble_models,
                                       n_neurons_ensemble_per_models=n_neurons_ensemble_per_models,
                                       fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                       lr=lr)

    models.append(ensemble)

    # ==========< Ensemble  CIP   >===============================================================================
    ensembleCIP = ensembleCIP_classification(name='Ensamble CIP',
                                       input_train=input_train, target_train=target_train,
                                       classes_labels=classes_labels,
                                       n_ensemble_models=n_ensemble_models,
                                       n_neurons_ensemble_per_models=n_neurons_ensemble_per_models,
                                       fn_activation=fn_activation1,
                                       beta=beta_cip, lamb=lamb_cip, lr=lr)

    models.append(ensembleCIP)

    # ==========< Ensemble  NCL   >==============================================================================
    ensembleNCL = ensembleNCL_classification(name='Ensamble NCL',
                                       input_train=input_train,
                                       classes_labels=classes_labels,
                                       n_ensemble_models=n_ensemble_models,
                                       n_neurons_ensemble_per_models=n_neurons_ensemble_per_models,
                                       fn_activation=fn_activation1,
                                       lamb=lamb_ncl, lr=lr)

    models.append(ensembleNCL)

    # ==========< MLP MSE  >======================================================================================
    netMLP = mlp_classification("MLP (%d neuronas)" % n_neurons_ensemble_per_models,
                                input_train, classes_labels,
                                n_neurons_ensemble_per_models,
                                fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                cost=mse, name_cost="MSE", lr=lr)

    models.append(netMLP)


    # ==========< MLP MSE MAX  >==================================================================================
    netMLP_MAX = mlp_classification("MLP (%d neuronas)" % (n_ensemble_models * n_neurons_ensemble_per_models),
                                input_train, classes_labels,
                                n_ensemble_models * n_neurons_ensemble_per_models,
                                fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                cost=mse, name_cost="MSE", lr=lr)

    models.append(netMLP_MAX)

    if only_cip:
        models = [ensembleCIP]

    plt.style.use('ggplot')

    # ============================================================================================================
    # Compile models and define extra score functions  >==========================================================

    if training:  # compile only if training models
        Logger().reset()
        for model in models:
            # Extra Scores
            if isinstance(model, EnsembleModel):
                model.append_score(score_ensemble_ambiguity, 'Ambigüedad')

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
                                  input_train=input_train, target_train=target_train,
                                  input_test=input_test, target_test=target_test,
                                  folds=folds, name_db=name_db, save_file=True, **args_train)
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

    plot_scores_classifications(models, input_train, target_train, input_test, target_test, classes_labels)

    plt.show()
