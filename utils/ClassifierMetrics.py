from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import numpy as np


class ClassifierMetrics:
    def __init__(self, target_names):
        self.metrics = []
        self.cm = []
        self.n = 0
        self.W = 3
        self.target_names = target_names

    def append(self, y_true, y_pred):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.target_names)
        self.cm.append(cm)

        TP = cm[0, 0]
        FP = np.sum(cm[1:, 0])
        FN = np.sum(cm[0, 1:])
        TN = np.sum(cm[1:, 1:])

        metrics = np.zeros((1, 4))
        metrics[0, 0] = TP / (TP + FP)
        metrics[0, 1] = TP / (TP + FN)
        metrics[0, 2] = FP / (FP + TN)
        metrics[0, 3] = TP ** 2 / ((TP + self.W * FP) * (TP + FN))

        if self.n == 0:
            self.metrics = metrics
        else:
            self.metrics = np.append(self.metrics, metrics, axis=0)
        self.n += 1

    def reset(self):
        self.metrics = []
        self.cm = []
        self.n = 0

    def print(self):
        if self.n == 1:
            print(self.cm)
        else:
            print(np.average(self.cm, axis=0))

        print("Results:")
        print("Precision:\t %f +- %f" % (np.mean(self.metrics[:, 0]), np.std(self.metrics[:, 0])))
        print("Recall:\t\t %f +- %f" % (np.mean(self.metrics[:, 1]), np.std(self.metrics[:, 1])))
        print("FPR:\t\t %f +- %f" % (np.mean(self.metrics[:, 2]), np.std(self.metrics[:, 2])))
        print("FoM:\t\t %f +- %f" % (np.mean(self.metrics[:, 3]), np.std(self.metrics[:, 3])))

    def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):

        if self.n == 1:
            cm = self.cm
        else:
            cm = np.average(self.cm, axis=0)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.target_names))
        plt.xticks(tick_marks, self.target_names, rotation=45)
        plt.yticks(tick_marks, self.target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


def test():
    metrics = ClassifierMetrics(3)
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    metrics.append(y_true, y_pred)
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 1, 2, 1, 0, 1]
    metrics.append(y_true, y_pred)
    metrics.print()


if __name__ == "__main__":
    test()
