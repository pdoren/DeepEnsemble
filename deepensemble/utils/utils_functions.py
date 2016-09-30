import theano.tensor as T


def entropy(px):
    return -T.sum(px * T.log(px))


def mutual_information(px1, px2, px1x2):
    return T.sum(px1x2 * (T.log(px1x2) - T.log(px1 * px2)))


def ambiguity(_input, model, ensemble):
    return T.power(model.output(_input) - ensemble.output(_input), 2.0)


def mean_ambiguity(_input, model, ensemble):
    return T.mean(ambiguity(_input, model, ensemble))


def bias(_input, ensemble, _target):
    sum_e = 0.0
    for model_j in ensemble.get_models():
        sum_e += (model_j.output(_input) - _target)

    return T.power(sum_e, 2.0)


def variance(_input, ensemble):
    sum_e = 0.0
    for model_j in ensemble.get_models():
        sum_e += (model_j.output(_input) - ensemble.output(_input))

    return T.power(sum_e, 2.0)
