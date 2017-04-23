import matplotlib.pylab as plt

from deepensemble.utils import cross_validation_score
from deepensemble.utils.cost_functions import mse, kullback_leibler_generalized
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils.utils_models import get_ensemble_model, get_ensembleCIP_model, \
    get_ensembleNCL_model, get_mlp_model


# noinspection PyDefaultArgument
def test_regression(name_db, data_input, data_target,
                    factor_number_neurons=0.75,
                    early_stop=True, no_update_best_parameters=False,
                    n_ensemble_models=4,
                    lamb_ncl=0.6,
                    beta_cip=0.6, lamb_cip=0.2, s=None, dist='CS',
                    cost_cip=mse, name_cost_cip='MSE', params_cost_cip={},
                    bias_layer=False, fn_activation1=ActivationFunctions.tanh,
                    fn_activation2=ActivationFunctions.sigmoid,
                    folds=10, lr_mse=0.01, lr_klg=0.001, max_epoch=300, batch_size=40):
    args_train = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': early_stop,
                  'improvement_threshold': 0.995, 'update_sets': True,
                  'no_update_best_parameters': no_update_best_parameters}

    #############################################################################################################
    # Define Parameters nets
    #############################################################################################################

    n_output = data_target.shape[1]
    n_inputs = data_input.shape[1]

    n_neurons_model = int(factor_number_neurons * (n_output + n_inputs))

    #############################################################################################################
    # Define Models
    #############################################################################################################

    models = []

    # ==========< Ensemble        >==============================================================================
    ensemble = get_ensemble_model(name='Ensamble',
                                  n_input=n_inputs, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_model,
                                  classification=False,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  cost=mse, name_cost="MSE",
                                  params_update={'learning_rate': lr_mse})

    models.append(ensemble)

    # ==========< Ensemble        >==============================================================================
    ensembleKLG = get_ensemble_model(name='Ensamble KLG',
                                     n_input=n_inputs, n_output=n_output,
                                     n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_model,
                                     classification=False,
                                     fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                     cost=kullback_leibler_generalized, name_cost="KLG",
                                     params_update={'learning_rate': lr_klg})

    models.append(ensembleKLG)

    # ==========< Ensemble  CIP   >===============================================================================
    ensembleCIP = get_ensembleCIP_model(name='Ensamble CIP',
                                        n_input=n_inputs, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=False,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        dist=dist,
                                        beta=beta_cip, lamb=lamb_cip, s=s, bias_layer=bias_layer, lr=lr_klg,
                                        cost=cost_cip, name_cost=name_cost_cip,
                                        params_cost=params_cost_cip,
                                        params_update={'learning_rate': lr_klg})

    models.append(ensembleCIP)

    # ==========< Ensemble  NCL   >==============================================================================
    ensembleNCL = get_ensembleNCL_model(name='Ensamble NCL',
                                        n_input=n_inputs, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=False,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        lamb=lamb_ncl, params_update={'learning_rate': lr_mse})

    models.append(ensembleNCL)

    # ==========< MLP MSE  >======================================================================================
    netMLP = get_mlp_model("MLP (%d neuronas)" % n_neurons_model,
                           n_input=n_inputs, n_output=n_output,
                           n_neurons=n_neurons_model,
                           classification=False,
                           fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                           cost=mse, name_cost="MSE", params_update={'learning_rate': lr_mse})

    models.append(netMLP)

    # ==========< MLP KLG  >======================================================================================
    netMLP_KLG = get_mlp_model("MLP KLG (%d neuronas)" % n_neurons_model,
                               n_input=n_inputs, n_output=n_output,
                               n_neurons=n_neurons_model,
                               classification=False,
                               fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                               cost=kullback_leibler_generalized, name_cost="KLG",
                               params_update={'learning_rate': lr_klg})

    models.append(netMLP_KLG)

    # ==========< MLP MSE MAX  >==================================================================================
    netMLP_MAX = get_mlp_model("MLP (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                               n_input=n_inputs, n_output=n_output,
                               n_neurons=n_ensemble_models * n_neurons_model,
                               classification=False,
                               fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                               cost=mse, name_cost="MSE", params_update={'learning_rate': lr_mse})

    models.append(netMLP_MAX)

    # ==========< MLP KLG MAX  >==================================================================================
    netMLP_KLG_MAX = get_mlp_model("MLP KLG (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                                   n_input=n_inputs, n_output=n_output,
                                   n_neurons=n_ensemble_models * n_neurons_model,
                                   classification=False,
                                   fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                   cost=kullback_leibler_generalized, name_cost="KLG",
                                   params_update={'learning_rate': lr_klg})

    models.append(netMLP_KLG_MAX)

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
