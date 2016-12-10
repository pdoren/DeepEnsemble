:mod:`deepensemble.metrics`

Metrics
=======

.. currentmodule:: deepensemble.metrics

Base Class Metrics
------------------

.. automodule:: deepensemble.metrics.basemetrics

.. autoclass:: BaseMetrics
   :members:

.. autoclass:: EnsembleMetrics
   :members:

.. autoclass:: FactoryMetrics
   :members:

Classifier Metrics
------------------

.. automodule:: deepensemble.metrics.classifiermetrics

.. autoclass:: ClassifierMetrics
   :members:

.. autoclass:: EnsembleClassifierMetrics
   :members:

Regression Metrics
------------------

.. automodule:: deepensemble.metrics.regressionmetrics

.. autoclass:: RegressionMetrics
   :members:

.. autoclass:: EnsembleMetrics
   :members:

Diversity
=========

.. automodule:: deepensemble.metrics.diversitymetrics

Utils
-----

.. autofunction:: oracle
.. autofunction:: contingency_table

Pairwise metrics
----------------

.. autofunction:: correlation_coefficient
.. autofunction:: disagreement_measure
.. autofunction:: double_fault_measure
.. autofunction:: kappa_statistic
.. autofunction:: q_statistic

Non Pairwise metrics
--------------------

.. autofunction:: kohavi_wolpert_variance
.. autofunction:: interrater_agreement
.. autofunction:: entropy_cc
.. autofunction:: entropy_sk
.. autofunction:: coincident_failure
.. autofunction:: difficulty
.. autofunction:: generalized_diversity