import theano
import numpy as np
from sklearn.datasets import fetch_mldata
from deepensemble.models.sequential import Sequential
from deepensemble.layers import *
from deepensemble.utils.cost_functions import *
from deepensemble.utils.utils_functions import ActivationFunctions
from sklearn import cross_validation
import matplotlib.pylab as plt

# Download Data
mnist = fetch_mldata("MNIST original", data_home="data")

data_input    = np.asarray(mnist.data.reshape(-1, 1, 28, 28), dtype=theano.config.floatX)
data_target   = ['%d' % i for i in mnist.target]
classes_names = ['%d' % i for i in range(10)]


# ## Training

nkerns = [5, 8]


net1 = Sequential("mlp", "classifier", classes_names)
net1.add_layer(Convolution2D(input_shape=(None, 1, 28, 28), num_filters=nkerns[0], filter_size=(5, 5)))
net1.add_layer(MaxPool2D(pool_size=(2, 2)))
net1.add_layer(Convolution2D(num_filters=nkerns[1], filter_size=(5, 5)))
net1.add_layer(MaxPool2D(pool_size=(2, 2)))
net1.add_layer(Dropout(p=0.5))
net1.add_layer(Dense(n_output=len(classes_names), activation=ActivationFunctions.softmax))
net1.append_cost(mse, name='Neg Log Likelihood')
net1.compile()

max_epoch = 5

input_train, input_test, target_train, target_test = cross_validation.train_test_split(
        data_input, data_target, test_size=0.3, random_state=0)

metrics = net1.fit(input_train, target_train, max_epoch=max_epoch, batch_size=200, early_stop=False)
# Compute metrics
metrics.append_prediction(input_test, target_test)


# ## Results
metrics.classification_report()
metrics.plot_confusion_matrix()
metrics.plot_cost(max_epoch, "Cost training")
metrics.plot_scores(max_epoch, "Accuracy training data")

plt.show()
