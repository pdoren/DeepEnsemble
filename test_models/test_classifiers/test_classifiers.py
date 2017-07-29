import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepensemble.metrics import EnsembleClassifierMetrics
from deepensemble.utils import cross_validation_score
from deepensemble.utils.cost_functions import mse
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils.utils_models import get_ensemble_model, get_ensembleCIP_model, \
    get_ensembleNCL_model, get_mlp_model
from deepensemble.utils.utils_test import load_model


# noinspection PyDefaultArgument
def test_classifiers(name_db, data_input, data_target, classes_labels,
                     factor_number_neurons=0.75,
                     is_binary=False, early_stop=False,
                     n_ensemble_models=4,
                     lamb_ncl=0.6,
                     is_cip_full=False,
                     beta_cip=0.9, lamb_cip=0.9,
                     s=None, dist='CS', bias_layer=False, mse_first_epoch=False,
                     annealing_enable=True,
                     fn_activation1=ActivationFunctions.sigmoid, fn_activation2=ActivationFunctions.sigmoid,
                     lr=0.01,
                     folds=10, max_epoch=300, batch_size=40):

    args_train_default = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': False,
                          'improvement_threshold': 0.995, 'update_sets': True}

    args_train_cip = {'max_epoch': max_epoch, 'batch_size': batch_size, 'early_stop': False,
                      'improvement_threshold': 0.995, 'update_sets': True, 'minibatch': True,
                      'criterion_update_params': 'cost', 'maximization_criterion': True}

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

    # ==========< Ensemble        >==============================================================================
    ensemble = get_ensemble_model(name='Ensamble',
                                  n_input=n_features, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_model,
                                  classification=True,
                                  classes_labels=classes_labels,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  cost=mse, name_cost="MSE",
                                  params_update={'learning_rate': lr})

    # ==========< Ensemble  CIP   >===============================================================================
    ensembleCIP = get_ensembleCIP_model(name='Ensamble CIP',
                                        n_input=n_features, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=True,
                                        is_cip_full=is_cip_full, mse_first_epoch=mse_first_epoch,
                                        classes_labels=classes_labels,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        annealing_enable=annealing_enable,
                                        dist=dist,
                                        beta=beta_cip, lamb=lamb_cip, s=s, bias_layer=bias_layer, lr=lr,
                                        params_update={'learning_rate': -lr})

    # ==========< Ensemble  NCL   >==============================================================================
    ensembleNCL = get_ensembleNCL_model(name='Ensamble NCL',
                                        n_input=n_features, n_output=n_output,
                                        n_ensemble_models=n_ensemble_models, n_neurons_models=n_neurons_model,
                                        classification=True,
                                        classes_labels=classes_labels,
                                        fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                        lamb=lamb_ncl, params_update={'learning_rate': lr})

    # ==========< MLP MSE MAX  >==================================================================================
    netMLP_MAX = get_mlp_model("MLP (%d neuronas)" % (n_ensemble_models * n_neurons_model),
                               n_input=n_features, n_output=n_output,
                               n_neurons=n_ensemble_models * n_neurons_model,
                               classification=True,
                               classes_labels=classes_labels,
                               fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                               cost=mse, name_cost="MSE", params_update={'learning_rate': lr})

    plt.style.use('fivethirtyeight')

    #############################################################################################################
    #
    #  TRAINING ALL MODELS
    #
    #############################################################################################################

    path_db = name_db + '/'

    models = []
    models.append(ensemble)
    models.append(ensembleNCL)
    models.append(netMLP_MAX)

    # noinspection PyUnusedLocal
    scores = cross_validation_score(models, data_input, data_target,
                                    folds=folds, path_db=path_db, **args_train_default)

    score_cip = cross_validation_score([ensembleCIP], data_input, data_target,
                                       folds=folds, path_db=path_db, **args_train_cip)

    scores[ensembleCIP.get_name()].append(score_cip)

    return scores


def show_data_classification(name_db, scores, max_epoch):
    plt.style.use('fivethirtyeight')
    r_score = {}
    for s in sorted(scores):
        d_score = scores[s]
        d = [(t1, t2) for t1, t2, _ in d_score]
        if "Ensamble" in s:
            d1 = [['%.4g, %.4g, %.4g' % f for f in t1.get_max_min_accuracy()] for _, _, t1 in d_score]
            d2 = [t1.get_fails() for _, _, t1 in d_score]
            print(s)
            print(d1)
            print(d2)

            _model = load_model(name_db, s)
            metrics = EnsembleClassifierMetrics(_model)
            for _, _, metric in d_score:
                metrics.append_metric(metric)
            metrics.plot_cost(title='Costo %s' % s, max_epoch=max_epoch)
            metrics.plot_costs(name=s, title='Costos %s' % s, max_epoch=max_epoch)
            metrics.plot_scores(title='Desempeño %s' % s, max_epoch=max_epoch)
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
