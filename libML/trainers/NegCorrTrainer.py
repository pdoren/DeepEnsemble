import theano.tensor as T
from theano import shared, config, function
import numpy as np
from collections import OrderedDict


class NegCorrTrainer:
    def __init__(self, classifiers, lamb=0.5, lr_adapt="ADADELTA", initial_learning_rate=1.0,
                 kernel_size=T.constant(1.0)):
        mlp_input = T.matrix('mlp_input')  # float matrix
        mlp_target = T.matrix('mlp_target')  # float vector
        # mlp_averaged_output = T.matrix('mlp_avg_output') # float vector

        N_output = classifiers[0].N_output
        self.classifiers = classifiers
        self.M = len(classifiers)
        self.mlp_output_single_proba = list()
        self.mlp_output_single_crisp = list()

        averaged_output_proba = 0.0
        for i in range(0, self.M):
            self.mlp_output_single_proba.append(function([mlp_input], classifiers[i].predict_proba(mlp_input)))
            if N_output < 2:
                self.mlp_output_single_crisp.append(function([mlp_input], classifiers[i].predict_binary(mlp_input)))
            else:
                self.mlp_output_single_crisp.append(function([mlp_input], classifiers[i].predict_multiclass(mlp_input)))
            averaged_output_proba += classifiers[i].predict_proba(mlp_input)
        averaged_output_proba = averaged_output_proba / self.M
        self.mlp_output_proba = function([mlp_input], averaged_output_proba)
        if N_output < 2:
            averaged_output_crisp = averaged_output_proba > 0.5
        else:
            averaged_output_crisp = T.argmax(averaged_output_proba, axis=1)
        self.mlp_output_crisp = function([mlp_input], averaged_output_crisp)
        reg_cons_L2 = T.scalar('lambdaL2', dtype=config.floatX)
        reg_cons_L1 = T.scalar('lambdaL1', dtype=config.floatX)
        rho = 0.95
        self.mlp_train = list()
        self.mlp_test = list()
        for i in range(0, self.M):
            cost_function = T.mean(T.power(classifiers[i].predict_proba(mlp_input) - mlp_target, 2.0)) - lamb * T.mean(
                T.power(classifiers[i].predict_proba(mlp_input) - averaged_output_proba, 2.0))
            cost_function += classifiers[i].sqr_L2(reg_cons_L2) + classifiers[i].L1(reg_cons_L1)
            updates = OrderedDict()
            gparams = [T.grad(cost_function, param) for param in classifiers[i].params]
            fudge_factor = 1e-6
            one = T.constant(1)
            for param, grad in zip(classifiers[i].params, gparams):
                value = param.get_value(borrow=True)
                accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
                delta_accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
                if lr_adapt is "ADAGRAD":
                    accu_new = accu + grad ** 2
                    updates[accu] = accu_new
                    updates[param] = param - (initial_learning_rate * grad / T.sqrt(accu_new + fudge_factor))
                elif lr_adapt is "ADADELTA":
                    # ADADELTA https://github.com/Lasagne/Lasagne/:
                    accu_new = rho * accu + (one - rho) * grad ** 2
                    updates[accu] = accu_new
                    update = (grad * T.sqrt(delta_accu + fudge_factor) / T.sqrt(accu_new + fudge_factor))
                    updates[param] = param - initial_learning_rate * update
                    delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
                    updates[delta_accu] = delta_accu_new
                elif lr_adapt is "CONS":
                    updates[param] = param - initial_learning_rate * grad
                else:
                    raise ValueError(
                        "Incorrect Learning rate adaptation scheme, valid options are CONS, ADAGRAD or ADADELTA")
            self.mlp_train.append(
                function([mlp_input, mlp_target, reg_cons_L2, reg_cons_L1], cost_function, updates=updates,
                         on_unused_input='ignore'))
            self.mlp_test.append(
                function([mlp_input, mlp_target, reg_cons_L2, reg_cons_L1], cost_function, on_unused_input='ignore'))

    def reset(self):
        for i in range(0, self.M):
            self.classifiers[i].reset()

    def minibach_eval(self, data, labels, reg_L1=0.0, reg_L2=0.0, batch_size=32, train=True):
        N = len(data)
        averaged_cost = np.zeros(shape=(self.M,))
        # avg_output = self.mlp_output_proba_average(data)
        for j in range(0, self.M):
            for NN, (start, end) in enumerate(
                    zip(range(0, len(data), batch_size), range(batch_size, len(data), batch_size))):
                if train:
                    averaged_cost[j] += self.mlp_train[j](data[start:end], labels[start:end],
                                                          reg_L2 * (end - start) / N, reg_L1 * (end - start) / N)
                else:
                    averaged_cost[j] += self.mlp_test[j](data[start:end], labels[start:end], reg_L2 * (end - start) / N,
                                                         reg_L1 * (end - start) / N)
        return averaged_cost / NN

    def compute_correlation_matrix(self, data):
        corr = np.ones(shape=(self.M, self.M))
        for i in range(0, self.M):
            Fi = self.mlp_output_single_proba[i](data) - np.mean(self.mlp_output_single_proba[i](data), axis=0)
            for j in range(i + 1, self.M):
                Fj = self.mlp_output_single_proba[j](data) - np.mean(self.mlp_output_single_proba[j](data), axis=0)
                corr[i, j] = np.trace(np.dot(Fi.T, Fj)) / np.sqrt(np.sum(np.power(Fi, 2.0)) * np.sum(np.power(Fj, 2.0)))
                corr[j, i] = corr[i, j]
        return corr
