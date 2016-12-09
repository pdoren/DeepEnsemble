:mod:`deepensemble.utils`

Utils Functions
===============

Cost Functions
--------------

.. automodule:: deepensemble.utils.cost_functions

.. autofunction:: mse
.. autofunction:: mcc
.. autofunction:: mee
.. autofunction:: cross_entropy
.. autofunction:: neg_log_likelihood
.. autofunction:: kullback_leibler
.. autofunction:: kullback_leibler_generalized
.. autofunction:: itakura_saito
.. autofunction:: dummy_cost

Cost Function for Ensemble Models
---------------------------------

.. autofunction:: neg_corr
.. autofunction:: neg_correntropy

Regularizer Functions
---------------------

.. automodule:: deepensemble.utils.regularizer_functions

.. autofunction:: L1
.. autofunction:: L2

Update Functions
----------------

.. automodule:: deepensemble.utils.update_functions

.. autofunction:: sgd
.. autofunction:: sgd_momentum
.. autofunction:: adadelta
.. autofunction:: adagrad
.. autofunction:: dummy_update

Score Functions
---------------

.. automodule:: deepensemble.utils.score_functions

.. autofunction:: score_accuracy
.. autofunction:: score_ensemble_ambiguity
.. autofunction:: score_rms
.. autofunction:: score_silverman
.. autofunction:: dummy_score

Logger
------

.. automodule:: deepensemble.utils.logger

.. autoclass:: Logger
   :members:

Utils Classifiers
-----------------

.. automodule:: deepensemble.utils.utils_classifiers

.. autofunction:: get_index_label_classes
.. autofunction:: translate_binary_target
.. autofunction:: translate_output
.. autofunction:: translate_target

Utils Data Bases
----------------

.. automodule:: deepensemble.utils.utils_data_bases

.. autofunction:: load_data
.. autofunction:: load_data_iris

Utils Functions
---------------

.. automodule:: deepensemble.utils.utils_functions

.. autoclass:: ActivationFunctions
   :members:

.. autoclass:: DiversityFunctions
   :members:

.. autoclass:: ITLFunctions
   :members:

Utils Models
------------

.. automodule:: deepensemble.utils.utils_models

.. autofunction:: ensemble_classification
.. autofunction:: ensembleCIP_classification
.. autofunction:: ensembleNCL_classification

Utils Testing
-------------

.. automodule:: deepensemble.utils.utils_test

.. autofunction:: plot_hist_train_test
.. autofunction:: plot_hist_train_test2
.. autofunction:: plot_scores_classifications
.. autofunction:: test_classifier
.. autofunction:: test_models
.. autofunction:: make_dirs


Serializable
============

.. automodule:: deepensemble.utils.serializable

.. autoclass:: Serializable
   :members:
