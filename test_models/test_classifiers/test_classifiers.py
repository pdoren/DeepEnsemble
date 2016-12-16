import matplotlib.pylab as plt

from deepensemble.utils.cost_functions import *
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils.utils_models import *
from deepensemble.utils import cross_validation_score, ITLFunctions


def test_classifiers(name_db, data_input, data_target, classes_labels,
                     is_binary=False, early_stop=True,
                     only_cip=False, n_ensemble_models = 4,
                     lamb_ncl=0.6, beta_cip=0.6, lamb_cip=0.2, s=None, kernel=ITLFunctions.kernel_gauss, dist='CS',
                     bias_layer=False,
                     fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                     folds=10, test_size=0.1, lr=0.01, max_epoch=300, batch_size=40):

    args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': early_stop, 'test_size': test_size,
                  'improvement_threshold': 0.995, 'update_sets': True, 'no_update_best_parameters': False}

    #############################################################################################################
    # Define Parameters nets
    #############################################################################################################

    n_features = data_input.shape[1]
    n_classes = len(classes_labels)

    n_output = 1 if is_binary and n_classes == 2 else n_classes
    n_inputs = n_features

    n_neurons = int(0.75 * (n_output + n_inputs))

    n_neurons_model = n_neurons

    #############################################################################################################
    # Define Models
    #############################################################################################################

    models = []

    # ==========< Ensemble        >==============================================================================
    ensemble = ensemble_classification(name='Ensamble',
                                       n_feature=n_features,
                                       classes_labels=classes_labels,
                                       n_ensemble_models=n_ensemble_models,
                                       n_neurons_model=n_neurons_model,
                                       fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                       lr=lr)

    models.append(ensemble)

    # ==========< Ensemble  CIP   >===============================================================================
    ensembleCIP = ensembleCIP_classification(name='Ensamble CIP',
                                             n_feature=n_features, classes_labels=classes_labels,
                                             n_ensemble_models=n_ensemble_models,
                                             n_neurons_models=n_neurons_model,
                                             kernel=kernel, dist=dist,
                                             fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                             beta=beta_cip, lamb=lamb_cip, s=s, bias_layer= bias_layer, lr=lr)

    models.append(ensembleCIP)

    # ==========< Ensemble  NCL   >==============================================================================
    ensembleNCL = ensembleNCL_classification(name='Ensamble NCL',
                                             input_train=n_features,
                                             classes_labels=classes_labels,
                                             n_ensemble_models=n_ensemble_models,
                                             n_neurons_models=n_neurons_model,
                                             fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                             lamb=lamb_ncl, lr=lr)

    models.append(ensembleNCL)

    # ==========< MLP MSE  >======================================================================================
    netMLP = mlp_classification("MLP (%d neuronas)" % n_neurons_model,
                                n_features, classes_labels,
                                n_neurons_model,
                                fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                cost=mse, name_cost="MSE", lr=lr)

    models.append(netMLP)

    # ==========< MLP MSE MAX  >==================================================================================
    netMLP_MAX = mlp_classification("MLP (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                                    n_features, classes_labels,
                                    n_ensemble_models * n_neurons_model,
                                    fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                    cost=mse, name_cost="MSE", lr=lr)

    models.append(netMLP_MAX)

    if only_cip:
        models = [ensembleCIP]

    plt.style.use('ggplot')

    #############################################################################################################
    #
    #  TRAINING ALL MODELS
    #
    #############################################################################################################

    path_db = name_db + '/'

    # noinspection PyUnusedLocal
    scores = cross_validation_score(models, data_input, data_target,
                                    folds=folds, path_db=path_db, **args_train)

    return scores