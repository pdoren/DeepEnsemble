class Trainer:
    def __init__(self, model):

        self.model = model

    def __eq__(self, other):
        if isinstance(other, Trainer):
            return self.model == other.model
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def trainer(self, input_train, target_train, input_test, target_test, max_epoch, validation_jump):
        pass
