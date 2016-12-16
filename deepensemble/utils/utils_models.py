from .update_functions import sgd
from .cost_functions import mse, cip_relevancy, cip_redundancy, neg_corr, cip_synergy, kullback_leibler_generalized
from .regularizer_functions import L2
from .logger import Logger
from .utils_functions import ITLFunctions

from ..combiner import PluralityVotingCombiner
from ..models import EnsembleModel, Sequential

__all__ = ["mlp_classification",
           "ensemble_classification", "ensembleCIP_classification", "ensembleNCL_classification"]


def _proc_pre_training(_ensemble, _input, _target, net0, batch_size, max_epoch):
    state_log = Logger().is_log_activate()
    Logger().log_disable()
    for net in _ensemble.get_models():
        net0.reset()
        net0.fit(_input, _target, batch_size=batch_size, max_epoch=max_epoch, early_stop=False)
        net.load_params(net0.save_params())
    if state_log:
        Logger().log_enable()

def mlp_classification(name,
                       n_feature, classes_labels,
                       n_neurons,
                       fn_activation1, fn_activation2,
                       bias_layer=False,
                       cost=mse, name_cost="MSE", lr=0.01):
    from ..layers.dense import Dense
    from ..layers.utils_layers import BiasLayer

    n_output = len(classes_labels)
    n_inputs = n_feature

    net = Sequential(name, "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons, activation=fn_activation1))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation2))
    if bias_layer:
        net.add_layer(BiasLayer(net, n_output=n_output))
    net.append_cost(cost, name=name_cost)
    net.set_update(sgd, name="SGD", learning_rate=lr)

    return net


def ensemble_classification(name,
                            n_feature, classes_labels,
                            n_ensemble_models, n_neurons_model,
                            fn_activation1, fn_activation2,
                            bias_layer=False,
                            cost=mse, name_cost="MSE", lr=0.01):
    ensemble = EnsembleModel(name=name)
    for i in range(n_ensemble_models):
        net = mlp_classification("net%d" % (i + 1),
                                 n_feature, classes_labels,
                                 n_neurons_model,
                                 fn_activation1, fn_activation2,
                                 bias_layer=bias_layer,
                                 cost=cost, name_cost=name_cost, lr=lr)
        ensemble.append_model(net)

    ensemble.set_combiner(PluralityVotingCombiner())

    return ensemble


def ensembleCIP_classification(name,
                               n_feature, classes_labels,
                               n_ensemble_models, n_neurons_models,
                               fn_activation1, fn_activation2,
                               dist='CS',
                               beta=0.9, lamb=0.9, s=None, kernel=ITLFunctions.kernel_gauss, bias_layer=True,
                               batch_size=40, max_epoch=300,
                               lr=0.01):
    ensemble = ensemble_classification(name,
                                       n_feature,
                                       classes_labels,
                                       n_ensemble_models, n_neurons_models,
                                       fn_activation1, fn_activation2,
                                       bias_layer= bias_layer,
                                       lr=lr)

    Logger().log_disable()
    net0 = mlp_classification("net0",
                              n_feature, classes_labels,
                              n_neurons_models,
                              fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                              lr=min(0.1, 5 * lr), cost=kullback_leibler_generalized, name_cost='KLG')
    net0.compile(fast=True)
    Logger().log_enable()

    ensemble.set_pre_training(proc_pre_training=_proc_pre_training,
                              params={'net0': net0, 'batch_size': batch_size, 'max_epoch': max_epoch})

    for net in ensemble.get_models():
        net.delete_cost('MSE')
        net.append_cost(cip_relevancy, name="CIP Relevancy", s=s, kernel=kernel, dist=dist)
        net.set_update(sgd, name="SGD", learning_rate=lr)

    ensemble.set_combiner(PluralityVotingCombiner())

    if beta != 0:
        ensemble.add_cost_ensemble(fun_cost=cip_redundancy, name="CIP Redundancy", beta=beta,
                                   s=s, kernel=kernel, dist=dist)

    if lamb != 0:
        ensemble.add_cost_ensemble(fun_cost=cip_synergy, name="CIP Synergy", lamb=lamb,
                                   s=s, kernel=kernel, dist=dist)

    return ensemble


def ensembleNCL_classification(name,
                               input_train,
                               classes_labels,
                               n_ensemble_models, n_neurons_models,
                               fn_activation1, fn_activation2,
                               lamb=0.6, lr=0.01, lamb_L2=0):
    ensemble = ensemble_classification(name,
                                       input_train,
                                       classes_labels,
                                       n_ensemble_models, n_neurons_models,
                                       fn_activation1, fn_activation2,
                                       lr=lr)

    if lamb_L2 > 0:
        for net in ensemble.get_models():
            net.append_reg(L2, name='Regularization L2', lamb=lamb_L2)

    ensemble.set_combiner(PluralityVotingCombiner())
    ensemble.add_cost_ensemble(fun_cost=neg_corr, name="NCL", lamb=lamb)

    return ensemble
