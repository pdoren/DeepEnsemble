import theano
import numpy as np
from sklearn.datasets import fetch_mldata
from deepensemble.models.sequential import Sequential
from deepensemble.layers.dense import Dense
from deepensemble.layers.conv import Convolution2D
from deepensemble.layers.pool import MaxPool2D
from deepensemble.utils.cost_functions import *
from deepensemble.utils.update_functions import *
from deepensemble.utils.utils_functions import ActivationFunctions
from sklearn import cross_validation
import matplotlib.pylab as plt

# Download Data
mnist = fetch_mldata("MNIST original", data_home="data")

data_input    = np.asarray(mnist.data, dtype=theano.config.floatX)
data_target   = mnist.target
classes_names = np.array(range(10))


# ## Training

nkerns = [20, 50]

net1 = Sequential("mlp", "classifier", classes_names)
net1.add_layer(Convolution2D(input_shape=data_input.shape, filter_shape=(nkerns[0], 1, 5, 5)))
net1.add_layer(MaxPool2D(pool_size=(2, 2)))
net1.add_layer(Convolution2D(filter_shape=(nkerns[1], nkerns[0], 5, 5)))
net1.add_layer(MaxPool2D(pool_size=(2, 2)))
net1.add_layer(Dense(n_output=len(classes_names), activation=ActivationFunctions.softmax))
net1.append_cost(neg_log_likelihood)
net1.set_update(sgd)
net1.compile()

max_epoch = 300

input_train, input_test, target_train, target_test = cross_validation.train_test_split(
        data_input, data_target, test_size=0.3, random_state=0)

metrics = net1.fit(input_train, target_train, max_epoch=max_epoch, batch_size=32, early_stop=False)
# Compute metrics
metrics.append_prediction(target_test, net1.predict(input_test))


# ## Results

metrics.classification_report()
metrics.plot_confusion_matrix()
metrics.plot_cost(max_epoch, "Cost training", log_yscale=True)
metrics.plot_score(max_epoch, "Accuracy training data")

plt.tight_layout()

