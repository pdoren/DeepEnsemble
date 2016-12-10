from sklearn import cross_validation
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.utils import load_data_cancer
from test_models.test_classifiers.test_classifiers import test_classifiers

SEED = 13

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data_cancer()

input_train, input_test, target_train, target_test = \
            cross_validation.train_test_split(data_input, data_target, test_size=0.3, stratify=data_target,
                                              random_state=SEED)

#############################################################################################################
# Testing
#############################################################################################################

test_classifiers(name_db, input_train, target_train, input_test, target_test, classes_labels,
                 only_cip=True,
                 lamb_ncl=0.6, beta_cip=0, lamb_cip=0, s=0.1,
                 fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                 folds=1, lr=0.02, training=True, max_epoch=500, batch_size=40)
