from .update_functions import sgd
from .cost_functions import mse, cip_relevancy, cip_redundancy, neg_corr, cip_synergy,\
    kullback_leibler_generalized, cip_full
from .regularizer_functions import L2
from .logger import Logger
from .utils_functions import ITLFunctions, ActivationFunctions
from .score_functions import mutual_information_cs

from ..combiner import AverageCombiner, PluralityVotingCombiner, SoftVotingCombiner,\
    WeightedVotingCombiner, SoftWeightVotingCombiner
from ..models import EnsembleModel, Sequential

__all__ = ["get_mlp_model",
           "get_ensemble_model", "get_ensembleCIP_model", "get_ensembleNCL_model", "get_ensembleCIP_KL_model"]


def _proc_pre_training(_ensemble, _input, _target, net0, batch_size, max_epoch):
    state_log = Logger().is_log_activate()
    Logger().log_disable()
    net0.reset()
    for net in _ensemble.get_models():
        net0.fit(_input, _target, batch_size=batch_size, max_epoch=max_epoch, early_stop=False)
        net.load_params(net0.save_params())
    if state_log:
        Logger().log_enable()


# noinspection PyDefaultArgument
def get_mlp_model(name,
                  n_input, n_output,
                  n_neurons,
                  fn_activation1, fn_activation2,
                  classes_labels=None,
                  classification=False,
                  bias_layer=False,
                  cost=mse, name_cost="MSE", params_cost={},
                  update=sgd, name_update='SGD', params_update={'learning_rate': 0.01}):
    from ..layers.dense import Dense
    from ..layers.utils_layers import BiasLayer

    if classification:
        net = Sequential(name, "classifier", classes_labels)
    else:
        net = Sequential(name)

    net.add_layer(Dense(n_input=n_input, n_output=n_neurons, activation=fn_activation1))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation2))
    if bias_layer:
        net.add_layer(BiasLayer(net, n_output=n_output))

    if cost is not None:
        net.append_cost(cost, name=name_cost, **params_cost)

    net.set_update(update, name=name_update, **params_update)

    return net


# noinspection PyDefaultArgument
def get_ensemble_model(name,
                       n_input, n_output,
                       n_ensemble_models, n_neurons_model,
                       fn_activation1, fn_activation2,
                       classes_labels=None,
                       classification=False,
                       bias_layer=False,
                       cost=mse, name_cost="MSE", params_cost={},
                       update=sgd, name_update='SGD', params_update={'learning_rate': 0.01}):
    ensemble = EnsembleModel(name=name)
    for i in range(n_ensemble_models):
        net = get_mlp_model("net%d" % (i + 1),
                            n_input=n_input, n_output=n_output,
                            n_neurons=n_neurons_model,
                            fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                            classes_labels=classes_labels,
                            classification=classification,
                            bias_layer=bias_layer,
                            cost=cost, name_cost=name_cost, params_cost=params_cost,
                            update=update, name_update=name_update, params_update=params_update)
        ensemble.append_model(net)

    if classification:
        ensemble.set_combiner(PluralityVotingCombiner())
    else:
        ensemble.set_combiner(AverageCombiner())

    ensemble.append_score(mutual_information_cs, name='Ics')

    return ensemble


# noinspection PyDefaultArgument
def get_ensembleCIP_model(name,
                          n_input, n_output,
                          n_ensemble_models, n_neurons_models,
                          fn_activation1, fn_activation2,
                          classes_labels=None,
                          classification=False,
                          dist='CS',
                          is_cip_full=False,
                          beta=0.9, lamb=0.9, s=None,
                          bias_layer=False,
                          batch_size=40, max_epoch=300,
                          cost=mse, name_cost="MSE", params_cost={}, lr=0.05,
                          update=sgd, name_update='SGD', params_update={'learning_rate': 0.01},
                          is_relevancy=True):

    if is_cip_full:
        cost_models = None
        name_cost_models = None
        params_cost_models = None
    else:
        cost_models = cip_relevancy
        name_cost_models = 'CIP Relevancy'
        params_cost_models = {'s': s, 'dist': dist}

    ensemble = get_ensemble_model(name,
                                  n_input=n_input, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_models,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  classes_labels=classes_labels,
                                  classification=classification,
                                  bias_layer=bias_layer,
                                  cost=cost_models, name_cost=name_cost_models,
                                  params_cost=params_cost_models,
                                  update=update, name_update=name_update, params_update=params_update)

    if not is_cip_full and is_relevancy and not bias_layer:
        Logger().log_disable()

        net0 = get_mlp_model("net0",
                             n_input=n_input, n_output=n_output,
                             n_neurons=n_neurons_models,
                             fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                             classification=classification,
                             classes_labels=classes_labels,
                             cost=cost, name_cost=name_cost, params_cost=params_cost,
                             params_update={'learning_rate': lr})
        net0.compile(fast=True)
        Logger().log_enable()

        ensemble.set_pre_training(proc_pre_training=_proc_pre_training,
                                  params={'net0': net0, 'batch_size': batch_size, 'max_epoch': max_epoch})

    if is_cip_full:
        ensemble.add_cost_ensemble(fun_cost=cip_full, name="CIP Full", s=s)
    else:
        if beta != 0:
            ensemble.add_cost_ensemble(fun_cost=cip_redundancy, name="CIP Redundancy", beta=beta,
                                       s=s, dist=dist)

        if lamb != 0:
            ensemble.add_cost_ensemble(fun_cost=cip_synergy, name="CIP Synergy", lamb=lamb,
                                       s=s, dist=dist)

    return ensemble


# noinspection PyDefaultArgument
def get_ensembleCIP_KL_model(name,
                              n_input, n_output,
                              n_ensemble_models, n_neurons_models,
                              fn_activation1, fn_activation2,
                              classes_labels=None,
                              classification=False,
                              dist='CS',
                              beta=0.9, lamb=0.9, s=None,
                              batch_size=40, max_epoch=300,
                              cost=mse, name_cost="MSE", params_cost={}, lr=0.05,
                              update=sgd, name_update='SGD', params_update={'learning_rate': 0.01},
                              is_relevancy=True):

    cost_models = kullback_leibler_generalized
    name_cost_models = 'KL'
    params_cost_models = {}

    ensemble = get_ensemble_model(name,
                                  n_input=n_input, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_models,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  classes_labels=classes_labels,
                                  classification=classification,
                                  cost=cost_models, name_cost=name_cost_models,
                                  params_cost=params_cost_models,
                                  update=update, name_update=name_update, params_update=params_update)

    if beta != 0:
        ensemble.add_cost_ensemble(fun_cost=cip_redundancy, name="CIP Redundancy", beta=beta,
                                   s=s, dist=dist)

    if lamb != 0:
        ensemble.add_cost_ensemble(fun_cost=cip_synergy, name="CIP Synergy", lamb=lamb,
                                   s=s, dist=dist)

    return ensemble


# noinspection PyDefaultArgument
def get_ensembleNCL_model(name,
                          n_input, n_output,
                          n_ensemble_models, n_neurons_models,
                          fn_activation1, fn_activation2,
                          classes_labels=None,
                          classification=False,
                          lamb=0.6, lamb_L2=0,
                          cost=mse, name_cost="MSE", params_cost={},
                          update=sgd, name_update='SGD', params_update={'learning_rate': 0.01}):
    ensemble = get_ensemble_model(name,
                                  n_input=n_input, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_models,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  classes_labels=classes_labels,
                                  classification=classification,
                                  cost=cost, name_cost=name_cost, params_cost=params_cost,
                                  update=update, name_update=name_update, params_update=params_update)

    if lamb_L2 > 0:
        for net in ensemble.get_models():
            net.append_reg(L2, name='Regularization L2', lamb=lamb_L2)

    ensemble.add_cost_ensemble(fun_cost=neg_corr, name="NCL", lamb=lamb)

    return ensemble
