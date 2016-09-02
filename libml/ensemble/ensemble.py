from libml.ensemble.pairmodeltrainer import PairModelTrainer
from libml.model import Model


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
            raise ValueError('No exist models in ensemble')


def test1():
    import theano
    import time
    import numpy as np
    import theano.tensor as T
    import matplotlib.pylab as plt
    from sklearn.datasets import load_iris
    from theano.sandbox import cuda
    from sklearn import cross_validation
    from utils.metrics.classifiermetrics import ClassifierMetrics
    from libml.nnet.mlp.mlpclassifier import MLPClassifier
    from libml.trainers.trainermlp import TrainerMLP
    from libml.ensemble.combiner.modelcombiner import ModelCombiner
    from libml.trainers.trainerensemble import TrainerEnsemble

    theano.config.floatX = 'float32'
    cuda.use('gpu')
    theano.config.compute_test_value = 'off'

    # Load Data

    iris = load_iris()

    data_input = np.asarray(iris.data, dtype=theano.config.floatX)
    data_target = iris.target_names[iris.target]
    classes_names = iris.target_names

    # Create models

    mlp1 = MLPClassifier(data_input.shape[1], [3], classes_names, output_activation=T.nnet.softmax,
                         hidden_activation=[T.tanh])

    mlp2 = MLPClassifier(data_input.shape[1], [3], classes_names, output_activation=T.nnet.softmax,
                         hidden_activation=[T.tanh])

    # Create Ensemble

    ensemble = Ensemble(ModelCombiner())

    trainerEnsemble = TrainerEnsemble(ensemble)

    ensemble.append_model(mlp1, TrainerMLP, cost="MSE", lr_adapt="CONS",
                          initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

    ensemble.append_model(mlp2, TrainerMLP, cost="MSE", lr_adapt="ADAGRAD",
                          initial_learning_rate=0.15, initial_momentum_rate=0.9, regularizer="L2+L1")

    # Generate data train and test
    max_epoch = 200
    validation_jump = 5

    input_train, input_test, target_train, target_test = cross_validation.train_test_split(
        data_input, data_target, test_size=0.4, random_state=0)

    # Initialize metrics
    metrics = ClassifierMetrics(classes_names)

    # Training

    tic = time.time()
    train_cost, test_cost, best_test_predict = \
        trainerEnsemble.trainer(input_train, target_train, input_test, target_test,
                                max_epoch=max_epoch, reg_L1=1e-2, reg_L2=1e-3, batch_size=32,
                                validation_jump=validation_jump, early_stop_th=4)
    toc = time.time()

    print("Elapsed time [s]: %f" % (toc - tic))

    # Compute metrics
    metrics.append_pred(target_test, ensemble.predict(input_test))
    metrics.append_cost(train_cost[:, 1], test_cost[:, 1])

    # Reset parameters
    ensemble.reset()

    metrics.print()
    metrics.plot_confusion_matrix()
    metrics.plot_cost(max_epoch)

    plt.show()

    print('TEST 1 OK')


if __name__ == "__main__":
    test1()
