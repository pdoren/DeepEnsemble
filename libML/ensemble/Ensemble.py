from libML.ensemble.Learner import Learner


class Ensemble:
    def __init__(self, combiner):
        self.combiner = combiner
        self.learners = []

    def append_learner(self, learner, class_trainer, **kwargs):

        new_learner = Learner(learner, class_trainer, **kwargs)
        if len(self.learners) == 0:
            self.learners.append(new_learner)
        elif self.learners[0] == new_learner:
            self.learners.append(new_learner)
        else:
            str_error = ''
            if self.learners[0].learner.N_input != learner.N_input:
                str_error += 'different input, '
            if self.learners[0].learner.N_output != learner.N_output:
                str_error += 'different output, '
            if self.learners[0].learner.type_learner is not learner.type_learner:
                str_error += 'different type learner, '

            raise ValueError('Incorrect Learner: ' + str_error[0:-2] + '.')


def test1():
    from libML.nnet.MLPClassifier import MLPClassifier
    from libML.nnet.MLPRegressor import MLPRegressor
    from libML.trainers.TrainerMLP import TrainerMLP
    import theano.tensor as T

    classes_names = ['class 1', 'class2']
    mlp1 = MLPClassifier(3, [5], classes_names,
                         output_activation=T.tanh,
                         hidden_activation=[T.tanh])

    mlp2 = MLPClassifier(3, [5, 2], classes_names,
                         output_activation=T.tanh,
                         hidden_activation=[T.tanh, T.tanh])

    mlp3 = MLPRegressor(3, [5, 2], 2, output_activation=T.tanh, hidden_activation=[T.tanh, T.tanh])

    ensemble = Ensemble([])

    ensemble.append_learner(mlp1, TrainerMLP, cost="MSE", lr_adapt="CONS",
                            initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

    ensemble.append_learner(mlp2, TrainerMLP, cost="MSE", lr_adapt="CONS",
                            initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

    ensemble.append_learner(mlp3, TrainerMLP, cost="MSE", lr_adapt="CONS",
                            initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

    print('TEST 1 OK')


if __name__ == "__main__":
    test1()
