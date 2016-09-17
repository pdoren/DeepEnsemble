import numpy as np

__all__ = ['oracle', 'contingency_table', 'disagreement_measure', 'double_fault_measure', 'q_statistic',
           'correlation_coefficient', 'kappa_statistic']

"""
Different metrics for compute diversity
"""


#
# Utils
#

def oracle(y, c):
    """ Compare between two arrays that represent the target and the output of a classifier.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c : numpy.array
        Output model for comparing with target.

    Notes
    -----
    The input arrays must be same shape and type (numpy.array).

    Returns
    -------
    numpy.array
    return an array with 1 and 0 by each elements of the input arrays.

    References
    ----------
    .. [1] Ludmila I. Kuncheva (2004), pp 298:
           Combining Pattern Classifiers Methods and Algorithms
           A Wiley-Interscience publication, ISBN 0-471-21078-1 (cloth).
    """
    if isinstance(y, np.array) and isinstance(c, np.array):
        raise ValueError("Incorrect type of arrays, they must be numpy.array")
    if y.shape != c.shape:
        raise ValueError("Incorrect arrays size")
    return (y == c).astype(int)


def contingency_table(y, c1, c2):
    """ Compute de contingency table.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Notes
    -----
    The input arrays must be same shape and type (numpy.array).

    Returns
    -------
    tuple
        Returns a tuple with the elements of contingency table between two classifiers.


    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    o1 = oracle(y, c1)
    o2 = oracle(y, c2)
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(o1)):
        if o1[i] == 1 and o2[i] == 1:
            a += 1
        elif o1[i] == 1 and o2[i] != 1:
            b += 1
        elif o1[i] != 1 and o2[i] == 1:
            c += 1
        else:
            d += 1
    return a, b, c, d


#
# Pairwise measures
#

def disagreement_measure(y, c1, c2):
    """ Measure is equal to the probability that the two classifiers will disagree on their decisions.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the disagreement measure between the classifiers 'c1' and 'c2', takes value in the range of [0, 1].

    References
    ----------
    .. [1] Skalak, D. (1996):
           The sources of increased accuracy for two proposed boosting algorithms. In Proc. American
           Association for Artificial Intelligence, AAAI-96, Integrating Multiple Learned Models Workshop.
    .. [2] Ludmila I. Kuncheva (2004), pp 300:
           Combining Pattern Classifiers Methods and Algorithms
           A Wiley-Interscience publication, ISBN 0-471-21078-1 (cloth).
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return (b + d) / (a + b + c + d)


def double_fault_measure(y, c1, c2):
    """ Measure is equal to the probability that both classifiers being wrong.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
    Return the double fault measure between the classifiers 'c1' and 'c2', takes value in the range of [0, 1].

    References
    ----------
    .. [1] Ludmila I. Kuncheva (2004), pp 301:
           Combining Pattern Classifiers Methods and Algorithms
           A Wiley-Interscience publication, ISBN 0-471-21078-1 (cloth).
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return d / (a + b + c + d)


def q_statistic(y, c1, c2):
    """ Q-Statistic.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the Q-Statistic measure between the classifiers 'c1' and 'c2'.
        Q-Statistic takes value in the range of [−1, 1]:

         - is zero if 'c1' and 'c2' are independent.
         - is positive if 'c1' and 'c2' make similar predictions.
         - is negative if 'c1' and 'c2' make different predictions.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return (a * d - b * c) / (a * d + b * c)


def correlation_coefficient(y, c1, c2):
    """ Correlation between two binary vectors in this case between the output of the classifiers.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the correlation coefficient between the output of the classifiers 'c1' and 'c2'.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return (a * d - b * c) / np.sqrt((a + b) * (a + c) * (c + d) * (b + d))


def kappa_statistic(y, c1, c2):
    """ Kappa-Statistic.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the Kappa-Statistic (kp) measure between the classifiers 'c1' and 'c2'.

         - kp = 1 if the two classifiers totally agree.
         - kp = 0 if the two classifiers agree by chance.
         - kp < 0 is a rare case where the agreement is even less than what is expected by chance.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    a, b, c, d = contingency_table(y, c1, c2)
    m = (a * d + b * c)
    theta1 = (a + d) / m
    theta2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (m * m)
    return (theta1 - theta2) / (1 - theta2)

#
# Non-PairwiseMeasures
#
