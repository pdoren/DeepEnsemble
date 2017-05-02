import os

import numpy as np


from deepensemble.utils import load_data, Serializable
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from test_models.test_classifiers.test_classifiers import test_classifiers, show_data_classification

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale',
                                                                              data_home='../../data', normalize=False)
y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(np.array(y)).eval()

#############################################################################################################
# Testing
#############################################################################################################

file_scores = name_db + '/score.pkl'

if not os.path.exists(file_scores):
    # 10-Cross Validation (sets: 90% train 10% test)
    scores = test_classifiers(name_db, data_input, data_target, classes_labels,
                              factor_number_neurons=1.0,
                              is_binary=False, early_stop=False,
                              n_ensemble_models=5,
                              lamb_ncl=0.8,
                              is_cip_full=False, bias_layer=False, mse_first_epoch=True,
                              annealing_enable=True,
                              beta_cip=0.3, lamb_cip=0.3, s=s, dist='CS',
                              fn_activation1=ActivationFunctions.sigmoid,
                              fn_activation2=ActivationFunctions.sigmoid,
                              folds=10, lr=0.1, max_epoch=300, batch_size=40)
    scores_data = Serializable(scores)
    scores_data.save(file_scores)
else:
    scores_data = Serializable()
    scores_data.load(file_scores)
    scores = scores_data.get_data()

show_data_classification(name_db, scores, 300)
