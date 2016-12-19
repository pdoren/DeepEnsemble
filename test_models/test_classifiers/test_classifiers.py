import matplotlib.pylab as plt

from deepensemble.utils.cost_functions import mse, kullback_leibler_generalized
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils.utils_models import get_ensemble_model, get_ensembleCIP_model,\
    get_ensembleNCL_model, get_mlp_model
from deepensemble.utils import cross_validation_score, ITLFunctions


def test_classifiers(name_db, data_input, data_target, classes_labels,
                     factor_number_neurons=0.75,
                     is_binary=False, early_stop=True, no_update_best_parameters=False,
                     only_cip=False, n_ensemble_models=4,
                     lamb_ncl=0.6,
                     beta_cip=0.6, lamb_cip=0.2, s=None, kernel=ITLFunctions.kernel_gauss, dist='CS',
                     cost_cip=mse, name_cost_cip='MSE', params_cost_cip={},
                     bias_layer=False, is_relevancy=False, pre_training=False,
                     fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                     folds=10, lr=0.01, max_epoch=300, batch_size=40):

    args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': early_stop,
                  'improvement_threshold': 0.995, 'update_sets': True,
                  'no_update_best_parameters': no_update_best_parameters}

    #############################################################################################################
    # Define Parameters nets
    #############################################################################################################

    n_features = data_input.shape[1]
    n_classes = len(classes_labels)

    n_output = 1 if is_binary and n_classes == 2 else n_classes
    n_inputs = n_features

    n_neurons_model = int(factor_number_neurons * (n_output + n_inputs))

    #############################################################################################################
    # Define Models
    #############################################################################################################

    models = []

    # ==========< Ensemble        >==============================================================================
    ensemble = get_ensemble_model(name='Ensamble',
                                  n_input=n_features, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_model,
                                  classification=True,
                                  classes_labels=classes_labels,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  cost=mse, name_cost="MSE",
                                  params_update={'learning_rate': lr})

    models.append(ensemble)

    # ==========< Ensemble        >==============================================================================
    ensembleKLG = get_ensemble_model(name='Ensamble KLG',
                                  n_input=n_features, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_model,
                                  classification=True,
                                  classes_labels=classes_labels,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  cost=kullback_leibler_generalized, name_cost="KLG",
                                  params_update={'learning_rate': lr})

    models.append(ensembleKLG)

    # ==========< Ensemble  CIP   >===============================================================================
    ensembleCIP = get_ensembleCIP_model(name='Ensamble CIP',
                                        n_input=n_features, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=True,
                                        classes_labels=classes_labels,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        kernel=kernel, dist=dist,
                                        beta=beta_cip, lamb=lamb_cip, s=s, bias_layer=bias_layer, lr=lr,
                                        is_relevancy=is_relevancy, pre_training=pre_training,
                                        cost=cost_cip, name_cost=name_cost_cip, params_cost=params_cost_cip,
                                        params_update={'learning_rate': lr})

    models.append(ensembleCIP)

    # ==========< Ensemble  NCL   >==============================================================================
    ensembleNCL = get_ensembleNCL_model(name='Ensamble NCL',
                                        n_input=n_features, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=True,
                                        classes_labels=classes_labels,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        lamb=lamb_ncl, params_update={'learning_rate': lr})

    models.append(ensembleNCL)

    # ==========< MLP MSE  >======================================================================================
    netMLP = get_mlp_model("MLP (%d neuronas)" % n_neurons_model,
                           n_input=n_features, n_output=n_output,
                           n_neurons=n_neurons_model,
                           classification=True,
                           classes_labels=classes_labels,
                           fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                           cost=mse, name_cost="MSE", params_update={'learning_rate': lr})

    models.append(netMLP)

    # ==========< MLP KLG  >======================================================================================
    netMLP_KLG = get_mlp_model("MLP KLG (%d neuronas)" % n_neurons_model,
                           n_input=n_features, n_output=n_output,
                           n_neurons=n_neurons_model,
                           classification=True,
                           classes_labels=classes_labels,
                           fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                           cost=kullback_leibler_generalized, name_cost="KLG", params_update={'learning_rate': lr})

    models.append(netMLP_KLG)

    # ==========< MLP MSE MAX  >==================================================================================
    netMLP_MAX = get_mlp_model("MLP (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                               n_input=n_features, n_output=n_output,
                               n_neurons=n_ensemble_models * n_neurons_model,
                               classification=True,
                               classes_labels=classes_labels,
                               fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                               cost=mse, name_cost="MSE", params_update={'learning_rate': lr})

    models.append(netMLP_MAX)

    # ==========< MLP KLG MAX  >==================================================================================
    netMLP_KLG_MAX = get_mlp_model("MLP KLG (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                                   n_input=n_features, n_output=n_output,
                                   n_neurons=n_ensemble_models * n_neurons_model,
                                   classification=True,
                                   classes_labels=classes_labels,
                                   fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                   cost=kullback_leibler_generalized, name_cost="KLG",
                                   params_update={'learning_rate': lr})

    models.append(netMLP_KLG_MAX)

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
