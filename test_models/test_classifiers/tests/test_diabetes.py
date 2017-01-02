import os

import numpy as np
from theano import shared

from deepensemble.utils import load_data, Serializable
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from test_models.test_classifiers.test_classifiers import test_classifiers, show_data_classification

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('diabetes_scale',
                                                                              data_home='../../data', normalize=False)
y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()

#############################################################################################################
# Testing
#############################################################################################################

file_scores = name_db + '/score.pkl'

if not os.path.exists(file_scores):
    # 10-Cross Validation (sets: 90% train 10% test)
    scores = test_classifiers(name_db, data_input, data_target, classes_labels,
                              factor_number_neurons=1.0,
                              is_binary=False, early_stop=False,
                              n_ensemble_models=3,
                              lamb_ncl=1.0,
                              beta_cip=1.0, lamb_cip=0.1, s=s, dist='CS',
                              fn_activation1=ActivationFunctions.sigmoid,
                              fn_activation2=ActivationFunctions.sigmoid,
                              folds=10, lr_mse=0.01, lr_klg=0.001, max_epoch=500, batch_size=40)
    scores_data = Serializable(scores)
    scores_data.save(file_scores)
else:
    scores_data = Serializable()
    scores_data.load(file_scores)
    scores = scores_data.get_data()


show_data_classification(name_db, scores, 500)
