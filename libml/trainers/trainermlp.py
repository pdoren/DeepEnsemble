import theano
import theano.tensor as T
import numpy as np
from .trainer import Trainer
from theano import shared, config, function
from collections import OrderedDict


class TrainerMLP(Trainer):

    def __init__(self, model, cost='MSE', regularizer=None, lr_adapt="ADADELTA",
                 initial_learning_rate=1.0, initial_momentum_rate=0.9,
                 kernel_size=T.constant(1.0), rho=0.95, fudge_factor=1e-6):

        super(TrainerMLP, self).__init__(model=model)

        mlp_input = T.matrix('mlp_input')  # float matrix
        mlp_target = T.matrix('mlp_target')  # float vector

        reg_cons_L2 = T.scalar('lambdaL2', dtype=config.floatX)
        reg_cons_L1 = T.scalar('lambdaL1', dtype=config.floatX)

        cost_function = model.get_cost_function(cost, mlp_input, mlp_target, kernel_size)

        if regularizer == "L1":
            cost_function += model.L1(reg_cons_L1)
        elif regularizer == "L2":
            cost_function += model.sqr_L2(reg_cons_L2)
        elif regularizer == "L2+L1":
            cost_function += model.sqr_L2(reg_cons_L2) + model.L1(reg_cons_L1)
        elif regularizer is None:
            pass
        else:
            raise ValueError("Incorrect regularizer, options are L1, L2, L2+L1 or None")

        updates = OrderedDict()
        gparams = [T.grad(cost_function, param) for param in model.params]

        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        for param, grad in zip(model.params, gparams):
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            delta_accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            velocity = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            if lr_adapt is "ADAGRAD":
                accu_new = accu + grad ** 2
                updates[accu] = accu_new
                updates[param] = param - (initial_learning_rate * grad / T.sqrt(accu_new + fudge_factor))
            elif lr_adapt is "ADADELTA":
                accu_new = rho * accu + (one - rho) * grad ** 2
                updates[accu] = accu_new
                update = (grad * T.sqrt(delta_accu + fudge_factor) / T.sqrt(accu_new + fudge_factor))
                updates[param] = param - initial_learning_rate * update
                delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
                updates[delta_accu] = delta_accu_new
            elif lr_adapt is "CONS":
                updates[param] = param - initial_learning_rate * grad
                x = initial_momentum_rate * velocity + updates[param]
                updates[velocity] = x - param
                updates[param] = x
            else:
                raise ValueError(
                    "Incorrect Learning rate adaptation scheme, valid options are CONS, ADAGRAD or ADADELTA")

        self.mlp_train = function([mlp_input, mlp_target, reg_cons_L1, reg_cons_L2], cost_function, updates=updates,
                                  on_unused_input='ignore')
        self.mlp_test = function([mlp_input, mlp_target, reg_cons_L1, reg_cons_L2], cost_function,
                                 on_unused_input='ignore')

    def minibatch_eval(self, data, labels, reg_L1=0.0, reg_L2=0.0, batch_size=32, train=True):

        averaged_cost = 0.0
        N = len(data)
        NN = 1
        for NN, (start, end) in enumerate(
                zip(range(0, len(data), batch_size), range(batch_size, len(data), batch_size))):
            if train:
                averaged_cost += self.mlp_train(data[start:end], labels[start:end], reg_L2 * (end - start) / N,
                                                reg_L1 * (end - start) / N)
            else:
                averaged_cost += self.mlp_test(data[start:end], labels[start:end], reg_L2 * (end - start) / N,
                                               reg_L1 * (end - start) / N)
        return averaged_cost / NN

    def trainer(self, input_train, target_train, input_test, target_test,
                max_epoch=100, validation_jump=5, reg_L1=0.01, reg_L2=0.01, batch_size=32, early_stop_th=4):

        target_train = self.model.translate_target(target_train)
        target_test = self.model.translate_target(target_test)

        train_cost = np.zeros(max_epoch)
        test_cost = np.zeros(int(max_epoch / validation_jump))

        test_cost[0] = self.minibatch_eval(input_test, target_test, reg_L2, reg_L1, batch_size, train=False)
        best_test_cost = test_cost[0]
        best_test_output = self.model.output(input_test)
        best_iteration = 0

        for epoch in range(0, max_epoch):
            # Present mini-batches in different order
            rand_perm = np.random.permutation(len(target_train))
            input_train = input_train[rand_perm]
            target_train = target_train[rand_perm]

            # Train minibatches
            train_cost[epoch] = self.minibatch_eval(input_train, target_train, reg_L2, reg_L1, batch_size,
                                                    train=True)
            # Early stopping
            if epoch > 0 and np.mod(epoch, validation_jump) == 0:
                val_ind = int(epoch / validation_jump)
                test_cost[val_ind] = self.minibatch_eval(input_test, target_test, reg_L2, reg_L1, batch_size,
                                                         train=False)
                if test_cost[val_ind] <= best_test_cost:
                    best_test_cost = test_cost[val_ind]
                    best_test_output = self.model.output(input_test)
                    best_iteration = epoch

                # Lutz Prechelt, Early stopping ... but when
                GL = 100 * (test_cost[val_ind] / best_test_cost - 1.0)  # generalization loss
                last_k_epoch = train_cost[epoch - validation_jump:epoch]
                Pk = 1000 * (np.sum(last_k_epoch) / (validation_jump * np.amin(last_k_epoch)) - 1.0)
                if GL / Pk > early_stop_th:
                    print("Early stopping at iter %d, best iter was %d" % (epoch, best_iteration))
                    # Fill cost functions with last value
                    test_cost[val_ind] = test_cost[val_ind]
                    train_cost[epoch] = train_cost[epoch]
                    break

        return train_cost, test_cost, self.model.translate_output(best_test_output)
