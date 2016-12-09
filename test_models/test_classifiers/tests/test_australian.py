from deepensemble.utils import load_data
from deepensemble.utils.utils_functions import ActivationFunctions
from test_models.test_classifiers.test_classifiers import test_classifiers

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale',
                                                                              data_home='../../data')

input_train = data_input[0:517]
input_test = data_input[518:690]
target_train = data_target[0:517]
target_test = data_target[518:690]


#############################################################################################################
# Testing
#############################################################################################################

test_classifiers(name_db, input_train, target_train, input_test, target_test, classes_labels,
                 only_cip=True,
                 lamb_ncl=0.6, beta_cip=0.4, lamb_cip=0.04,
                 fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                 folds=5, lr=0.02, training=True, max_epoch=500, batch_size=50)
