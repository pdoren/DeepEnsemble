import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from deepensemble.utils.utils_test import load_model
from deepensemble.metrics import EnsembleClassifierMetrics
from deepensemble.utils.cost_functions import mse, kullback_leibler_generalized
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils.utils_models import get_ensemble_model, get_ensembleCIP_model, \
    get_ensembleNCL_model, get_mlp_model
from deepensemble.utils import cross_validation_score, ITLFunctions


# noinspection PyDefaultArgument
def test_classifiers(name_db, data_input, data_target, classes_labels,
                     factor_number_neurons=0.75,
                     is_binary=False, early_stop=True, no_update_best_parameters=False,
                     n_ensemble_models=4,
                     lamb_ncl=0.6,
                     beta_cip=0.6, lamb_cip=0.2, s=None, kernel=ITLFunctions.kernel_gauss, dist='CS',
                     cost_cip=mse, name_cost_cip='MSE', params_cost_cip={},
                     bias_layer=False,
                     fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                     lr_mse=0.01, lr_klg=0.001,
                     folds=10, max_epoch=300, batch_size=40):
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
                                  params_update={'learning_rate': lr_mse})

    models.append(ensemble)

    # ==========< Ensemble        >==============================================================================
    ensembleKLG = get_ensemble_model(name='Ensamble KLG',
                                     n_input=n_features, n_output=n_output,
                                     n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_model,
                                     classification=True,
                                     classes_labels=classes_labels,
                                     fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                     cost=kullback_leibler_generalized, name_cost="KLG",
                                     params_update={'learning_rate': lr_klg})

    models.append(ensembleKLG)

    # ==========< Ensemble  CIP   >===============================================================================
    ensembleCIP = get_ensembleCIP_model(name='Ensamble CIP',
                                        n_input=n_features, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=True,
                                        classes_labels=classes_labels,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        kernel=kernel, dist=dist,
                                        beta=beta_cip, lamb=lamb_cip, s=s, bias_layer=bias_layer, lr=lr_klg,
                                        cost=cost_cip, name_cost=name_cost_cip, params_cost=params_cost_cip,
                                        params_update={'learning_rate': lr_klg})

    models.append(ensembleCIP)

    # ==========< Ensemble  CIP KL  >=============================================================================
    ensembleCIP_KL = get_ensembleCIP_model(name='Ensamble CIP KL',
                                           n_input=n_features, n_output=n_output,
                                           n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                           classification=True,
                                           classes_labels=classes_labels,
                                           fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                           kernel=kernel, dist=dist,
                                           is_relevancy=False,
                                           beta=beta_cip, lamb=lamb_cip, s=s, bias_layer=bias_layer, lr=lr_klg,
                                           cost=cost_cip, name_cost=name_cost_cip, params_cost=params_cost_cip,
                                           params_update={'learning_rate': lr_klg})

    models.append(ensembleCIP_KL)

    # ==========< Ensemble  NCL   >==============================================================================
    ensembleNCL = get_ensembleNCL_model(name='Ensamble NCL',
                                        n_input=n_features, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=True,
                                        classes_labels=classes_labels,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        lamb=lamb_ncl, params_update={'learning_rate': lr_mse})

    models.append(ensembleNCL)

    # ==========< MLP MSE  >======================================================================================
    netMLP = get_mlp_model("MLP (%d neuronas)" % n_neurons_model,
                           n_input=n_features, n_output=n_output,
                           n_neurons=n_neurons_model,
                           classification=True,
                           classes_labels=classes_labels,
                           fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                           cost=mse, name_cost="MSE", params_update={'learning_rate': lr_mse})

    models.append(netMLP)

    # ==========< MLP KLG  >======================================================================================
    netMLP_KLG = get_mlp_model("MLP KLG (%d neuronas)" % n_neurons_model,
                               n_input=n_features, n_output=n_output,
                               n_neurons=n_neurons_model,
                               classification=True,
                               classes_labels=classes_labels,
                               fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                               cost=kullback_leibler_generalized, name_cost="KLG",
                               params_update={'learning_rate': lr_klg})

    models.append(netMLP_KLG)

    # ==========< MLP MSE MAX  >==================================================================================
    netMLP_MAX = get_mlp_model("MLP (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                               n_input=n_features, n_output=n_output,
                               n_neurons=n_ensemble_models * n_neurons_model,
                               classification=True,
                               classes_labels=classes_labels,
                               fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                               cost=mse, name_cost="MSE", params_update={'learning_rate': lr_mse})

    models.append(netMLP_MAX)

    # ==========< MLP KLG MAX  >==================================================================================
    netMLP_KLG_MAX = get_mlp_model("MLP KLG (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                                   n_input=n_features, n_output=n_output,
                                   n_neurons=n_ensemble_models * n_neurons_model,
                                   classification=True,
                                   classes_labels=classes_labels,
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


def show_data(name_db, scores):
    plt.style.use('ggplot')
    r_score = {}
    d_diversity = {}
    for s in scores:
        d_score = scores[s]
        d = [(t1, t2) for t1, t2, _ in d_score]
        if "Ensamble" in s:
            metrics = [t1.get_fails() for _, _, t1 in d_score]
            print(s)
            print(metrics)
        if "Ensamble CIP" in s:
            _model = load_model(name_db, s)
            metrics = EnsembleClassifierMetrics(_model)
            for _, _, metric in d_score:
                metrics.append_metric(metric)
            metrics.plot_cost(title='Costo %s' % s, max_epoch=300)
            metrics.plot_costs(name=s, title='Costos %s' % s, max_epoch=300)
            metrics.plot_scores(title='Desempeño %s' % s, max_epoch=300)
        _mean = np.mean(d, axis=0)
        _std = np.std(d, axis=0)
        max_score = np.max(d, axis=0)
        min_score = np.min(d, axis=0)
        s1 = ['%.2f +-%.2f' % (100 * _mean[0], 100 * _std[0])]
        s2 = ['%.2f +-%.2f' % (100 * _mean[1], 100 * _std[1])]
        s3 = ['%.2f / %.2f' % (100 * max_score[1], 100 * min_score[1])]
        r_score[s] = [s1, s2, s3]

    df = pd.DataFrame(r_score)

    print(df)

    plt.show()
