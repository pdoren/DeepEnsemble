from libml.model import Model
from libml.ensemble.pairmodeltrainer import PairModelTrainer


class Ensemble(Model):
    def __init__(self, combiner):
        super(Ensemble, self).__init__(n_input=0, n_output=0)
        self.combiner = combiner
        self.list_models_ensemble = []

    def append_model(self, model, class_trainer, **kwargs):

        new_model = PairModelTrainer(model, class_trainer, **kwargs)
        if len(self.list_models_ensemble) == 0:
            # copy data model
            self.n_input = new_model.model.n_input
            self.n_output = new_model.model.n_output
            self.type_model = new_model.model.type_model
            self.target_labels = new_model.model.target_labels

            self.list_models_ensemble.append(new_model)
        elif self.list_models_ensemble[0] == new_model:
            self.list_models_ensemble.append(new_model)
        else:
            str_error = ''
            if self.list_models_ensemble[0].model.N_input != model.N_input:
                str_error += 'different input, '
            if self.list_models_ensemble[0].model.N_output != model.N_output:
                str_error += 'different output, '
            if self.list_models_ensemble[0].model.type_learner is not model.type_learner:
                str_error += 'different type learner, '
            if self.list_models_ensemble[0].model.target_labels is not model.target_labels:
                str_error += 'different target labels, '

            raise ValueError('Incorrect Learner: ' + str_error[0:-2] + '.')

    def reset(self):
        for model in self.list_models_ensemble:
            model.reset()

    def output(self, _input):
        return self.combiner.output(self.list_models_ensemble, _input)

    def translate_output(self, _output):
        if len(self.list_models_ensemble) != 0:
            return self.list_models_ensemble[0].model.translate_output(_output)
        else:
            raise ValueError('')


def test1():
    from libml.nnet.mlpclassifier import MLPClassifier
    from libml.trainers.trainermlp import TrainerMLP
    from libml.ensemble.combiner.modelcombiner import ModelCombiner
    import theano
    import numpy as np
    import theano.tensor as T

    classes_names = np.array(['class 1', 'class2'])
    mlp1 = MLPClassifier(3, [5], classes_names, output_activation=T.tanh, hidden_activation=[T.tanh])

    mlp2 = MLPClassifier(3, [5, 2], classes_names, output_activation=T.tanh, hidden_activation=[T.tanh, T.tanh])

    ensemble = Ensemble(ModelCombiner())

    ensemble.append_model(mlp1, TrainerMLP, cost="MSE", lr_adapt="CONS",
                          initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

    ensemble.append_model(mlp2, TrainerMLP, cost="MSE", lr_adapt="CONS",
                          initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

    ensemble.reset()
    _input = np.asarray([[1, 1, 1], [1, 0, 5]], dtype=theano.config.floatX)
    print('Prediction: ', ensemble.predict(_input))

    print('TEST 1 OK')


if __name__ == "__main__":
    test1()
