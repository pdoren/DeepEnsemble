class Trainer:
    def __init__(self, learner):

        self.learner = learner

    def __eq__(self, other):
        if isinstance(other, Trainer):
            return self.learner == other.learner
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def trainer(self, input_train, target_train, input_test, target_test):
        pass
