from unittest import TestCase

__author__ = 'pdoren'
__project__ = 'DeepEnsemble'


class TestITLFunctions(TestCase):
    def test_cross_information_potential(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        from theano import shared
        import numpy as np
        import theano.tensor as T
        from sklearn.metrics import mutual_info_score

        y1 = np.array([1, 1, 0, 0, 1, 1, 0])
        y2 = np.array([0, 0, 1, 1, 0, 0, 1])

        Y = [y1[:, np.newaxis], y2[:, np.newaxis]]

        s = 1.06 * np.std(y1) * (len(y1)) ** (-0.2)

        Ics = -T.log(ITLFunctions.cross_information_potential(Y, kernel=ITLFunctions.kernel_gauss, s=s))
        I = mutual_info_score(y1, y2)

        self.assertEqual(Ics.eval(), I, 'Problem with name of class')

    def test_mutual_information_ed(self):
        self.fail()
