from .update_functions import sgd
from .cost_functions import mse, cip2, cip_diversity, neg_corr, cauchy_schwarz_divergence
from .regularizer_functions import L2
from .utils_functions import ActivationFunctions
from .logger import Logger

from ..combiner import PluralityVotingCombiner, GeometricVotingCombiner
from ..models import EnsembleModel, Sequential

__all__ = ["mlp_classification",
    "ensemble_classification", "ensembleCIP_classification", "ensembleNCL_classification"]



def _proc_pre_training(_ensemble, _input, _target, net0, batch_size, max_epoch):
    Logger().log_disable()
    for net in _ensemble.get_models():
        net0.fit(_input, _target, batch_size=batch_size, max_epoch=max_epoch, early_stop=False)
        net.load_params(net0.save_params())
        net0.reset()
    Logger().log_enable()


def mlp_classification(name,
                       input_train, classes_labels,
                       n_neurons,
                       fn_activation1, fn_activation2,
                       cost=mse, name_cost="MSE", lr=0.01):

    from ..layers.dense import Dense

    n_output = len(classes_labels)
    n_inputs = input_train.shape[1]

    net = Sequential(name, "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_inputs, n_output=n_neurons, activation=fn_activation1))
    net.add_layer(Dense(n_output=n_output, activation=fn_activation2))
    net.append_cost(cost, name=name_cost)
    net.set_update(sgd, name="SGD", learning_rate=lr)

    return net


def ensemble_classification(name,
                            input_train, classes_labels,
                            n_ensemble_models, n_neurons_ensemble_per_models,
                            fn_activation1, fn_activation2,
                            cost=mse, name_cost="MSE", lr=0.01):

    ensemble = EnsembleModel(name=name)
    for i in range(n_ensemble_models):
        net = mlp_classification("net%d" % (i + 1),
                                 input_train, classes_labels,
                                 n_neurons_ensemble_per_models,
                                 fn_activation1, fn_activation2,
                                 cost=cost, name_cost=name_cost, lr=lr)
        ensemble.append_model(net)

    ensemble.set_combiner(PluralityVotingCombiner())

    return ensemble


def ensembleCIP_classification(name,
                               input_train, target_train,
                               classes_labels,
                               n_ensemble_models, n_neurons_ensemble_per_models, fn_activation,
                               beta=0.9, batch_size=40, max_epoch=300,
                               lr=0.01):

    ensemble = ensemble_classification(name,
                                       input_train,
                                       classes_labels,
                                       n_ensemble_models, n_neurons_ensemble_per_models,
                                       fn_activation, ActivationFunctions.softmax,
                                       lr=lr)

    Logger().log_disable()
    net0 = mlp_classification("net0",
                              input_train, classes_labels,
                              n_neurons_ensemble_per_models,
                              fn_activation1=fn_activation, fn_activation2=ActivationFunctions.softmax,
                              lr=min(0.1, 10 * lr))
    net0.compile(fast=True)
    Logger().log_enable()

    ensemble.set_pre_training(proc_pre_training=_proc_pre_training,
                              params={'net0': net0, 'batch_size': batch_size, 'max_epoch': max_epoch})

    for net in ensemble.get_models():
        net.append_cost(cip2, name="CIP")
        net.set_update(sgd, name="SGD", learning_rate=lr)

    ensemble.set_combiner(GeometricVotingCombiner())

    if beta > 0:
        ensemble.add_cost_ensemble(fun_cost=cip_diversity, name="CIP Diversity", beta=beta)

    return ensemble


def ensembleNCL_classification(name,
                               input_train,
                               classes_labels,
                               n_ensemble_models, n_neurons_ensemble_per_models, fn_activation,
                               lamb=0.6, lr=0.01, lamb_L2=0):
    ensemble = ensemble_classification(name,
                                       input_train,
                                       classes_labels,
                                       n_ensemble_models, n_neurons_ensemble_per_models,
                                       fn_activation, ActivationFunctions.sigmoid,
                                       lr=lr)

    if lamb_L2 > 0:
        for net in ensemble.get_models():
            net.append_reg(L2, name='Regularization L2', lamb=lamb_L2)

    ensemble.set_combiner(PluralityVotingCombiner())
    ensemble.add_cost_ensemble(fun_cost=neg_corr, name="NCL", lamb=lamb)

    return ensemble
