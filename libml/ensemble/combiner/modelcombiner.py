import theano.tensor as T


class ModelCombiner:
    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def output(self, list_models_ensemble, _input):
        output = 0.0
        for pair in list_models_ensemble:
            output += pair.model.output(_input)
        n = T.constant(len(list_models_ensemble))
        return output / n
