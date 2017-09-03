from theano import shared

from .cost_functions import mse, cip_relevancy, cip_redundancy, neg_corr, cip_synergy, cip_full
from .logger import Logger
from .regularizer_functions import L2
from .score_functions import mutual_information_cs
from .update_functions import sgd, count_iterations, sgd_cip
from .utils_functions import ITLFunctions
from ..combiner import AverageCombiner, PluralityVotingCombiner
from ..models import EnsembleModel, Sequential
from ..utils.utils_translation import TextTranslation

__all__ = ["get_mlp_model",
           "get_ensemble_model",
           "get_ensembleCIP_model",
           "get_ensembleNCL_model"
           ]


def _proc_pre_training(_ensemble, _input, _target, net0, batch_size, max_epoch):
    state_log = Logger().is_log_activate()
    Logger().log_disable()

    for net in _ensemble.get_models():
        net0.reset()
        net0.fit(_input, _target, batch_size=batch_size, max_epoch=max_epoch, early_stop=False)
        params = net0.save_params()
        net.load_params(params)

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

    if update is not None:
        net.set_update(update, name=name_update, **params_update)

    return net


# noinspection PyDefaultArgument
def get_ensemble_model(name,
                       n_input, n_output,
                       n_ensemble_models, n_neurons_model,
                       fn_activation1, fn_activation2,
                       classes_labels=None,
                       classification=False,
                       fn_combiner=None,
                       args_combiner={},
                       bias_layer=False,
                       cost=mse, name_cost="MSE", params_cost={},
                       update=sgd, name_update='SGD', params_update={'learning_rate': 0.01},
                       list_scores=[{'fun_score': mutual_information_cs, 'name': TextTranslation().get_str('MI')}]):
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

    if fn_combiner is None:
        if classification:
            ensemble.set_combiner(PluralityVotingCombiner())
        else:
            ensemble.set_combiner(AverageCombiner())
    else:
        ensemble.set_combiner(fn_combiner(**args_combiner))

    for score in list_scores:
        ensemble.append_score(score['fun_score'], score['name'])


    return ensemble


# noinspection PyDefaultArgument
def get_ensembleCIP_model(name,
                          n_input, n_output,
                          n_ensemble_models, n_neurons_models,
                          fn_activation1, fn_activation2,
                          classes_labels=None,
                          classification=False,
                          fn_combiner=None,
                          args_combiner={},
                          dist='CS',
                          is_cip_full=False,
                          beta=0.9, lamb=0.9, s=None, lsp=1.5, lsm=0.5,
                          bias_layer=False, mse_first_epoch=False,
                          batch_size=40, max_epoch=300,
                          cost=mse, name_cost="MSE", params_cost={}, lr=0.01, annealing_enable=False,
                          update=sgd, name_update='SGD', params_update={'learning_rate': 0.01}, type='',
                          list_scores=[{'fun_score': mutual_information_cs, 'name': TextTranslation().get_str('MI')}]):
    if annealing_enable:
        current_epoch = shared(0, 'current_epoch')  # count current epoch
        si = ITLFunctions.annealing(lsp * s, lsm * s, current_epoch, max_epoch * batch_size)
    else:
        si = s

    update_models = update
    if is_cip_full:
        cost_models = None
        name_cost_models = None
        params_cost_models = None
        update_models = None
    else:
        cost_models = cip_relevancy
        name_cost_models = TextTranslation().get_str('Relevancy') + ' CIP(%s)' % dist
        params_cost_models = {'s': si, 'dist': dist, 'type': type}

    ensemble = get_ensemble_model(name,
                                  n_input=n_input, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_models,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  classes_labels=classes_labels,
                                  classification=classification,
                                  fn_combiner=fn_combiner,
                                  args_combiner=args_combiner,
                                  bias_layer=bias_layer,
                                  cost=cost_models, name_cost=name_cost_models,
                                  params_cost=params_cost_models,
                                  update=update_models, name_update=name_update, params_update=params_update,
                                  list_scores=list_scores)

    if mse_first_epoch:
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
                                  params={'net0': net0, 'batch_size': batch_size,
                                          'max_epoch': max(200, int(0.2 * max_epoch))})

    if is_cip_full:
        ensemble.append_cost(fun_cost=cip_full, name="CIP Full", s=s, dist=dist, type=type)
    elif n_ensemble_models != 1:
        if beta != 0:
            ensemble.add_cost_ensemble(fun_cost=cip_redundancy, name=TextTranslation().get_str('Redundancy') + ' CIP(%s)' % dist,
                                       beta=-beta, s=s, dist=dist, type=type)

        if lamb != 0:
            ensemble.add_cost_ensemble(fun_cost=cip_synergy, name=TextTranslation().get_str('Synergy') + ' CIP(%s)' % dist,
                                       lamb=lamb, s=s, dist=dist, type=type)

    if update == sgd_cip:
        ensemble.update_io()
        params_update['error'] = ensemble.get_error(prob=True)
    ensemble.set_update(update, name=name_update, **params_update)

    if annealing_enable:
        # noinspection PyUnboundLocalVariable
        ensemble.append_update(count_iterations, 'Count Epoch', _i=current_epoch)

    return ensemble


# noinspection PyDefaultArgument
def get_ensembleNCL_model(name,
                          n_input, n_output,
                          n_ensemble_models, n_neurons_models,
                          fn_activation1, fn_activation2,
                          classes_labels=None,
                          classification=False,
                          fn_combiner=None,
                          args_combiner={},
                          lamb=0.6, lamb_L2=0,
                          cost=mse, name_cost="MSE", params_cost={},
                          update=sgd, name_update='SGD', params_update={'learning_rate': 0.01}):
    ensemble = get_ensemble_model(name,
                                  n_input=n_input, n_output=n_output,
                                  n_ensemble_models=n_ensemble_models, n_neurons_model=n_neurons_models,
                                  fn_activation1=fn_activation1, fn_activation2=fn_activation2,
                                  classes_labels=classes_labels,
                                  classification=classification,
                                  fn_combiner=fn_combiner,
                                  args_combiner=args_combiner,
                                  cost=cost, name_cost=name_cost, params_cost=params_cost,
                                  update=update, name_update=name_update, params_update=params_update)

    if lamb_L2 > 0:
        for net in ensemble.get_models():
            net.append_reg(L2, name=TextTranslation().get_str('Regularization') + ' L2', lamb=lamb_L2)

    ensemble.add_cost_ensemble(fun_cost=neg_corr, name="NCL", lamb=lamb)

    return ensemble
