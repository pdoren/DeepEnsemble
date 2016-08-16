import numpy as np
from .trainer import Trainer


class TrainerEnsemble(Trainer):
    def __init__(self, model):
        super(TrainerEnsemble, self).__init__(model=model)

    def trainer(self, input_train, target_train, input_test, target_test, max_epoch, validation_jump, **kwargs):
        n_models = len(self.model.list_models_ensemble)
        train_cost = np.zeros(shape=(max_epoch, n_models))
        test_cost = np.zeros(shape=(int(max_epoch / validation_jump), n_models))
        best_test_output = []
        for i, pair in enumerate(self.model.list_models_ensemble):
            trc, tec, bto = pair.trainer.trainer(input_train, target_train, input_test, target_test, max_epoch, **kwargs)
            train_cost[:, i] = trc
            test_cost[:, i] = tec
            best_test_output.append(bto)

        return train_cost, test_cost, best_test_output
