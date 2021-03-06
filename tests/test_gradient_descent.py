import numpy as np
import unittest

from algorithms.gradient_descent import GradientDescent
import utils.constants as constants


class TestGradientDescent(unittest.TestCase):
    """A unit test for algorithms.GradientDescent.
    """
    def test_insert_zero(self):
        """A unit test of GradientDescent()._insert_zero.
        """
        x = np.asarray([[2], [3], [1], [2], [1], [4], [5], [1]])
        y = [2, 3, 0, 2, 0, 4, 5, 0]
        lambda_step = 2
        indices_removed = np.ravel(
            np.delete(np.argwhere(np.abs(x) < lambda_step), 1, 1))
        x_updated = GradientDescent()._insert_zero(y, indices_removed)
        x_expected = np.asarray([2, 3, 0, 0, 0, 2, 0, 0, 4, 5, 0])
        np.testing.assert_array_equal(x_updated, x_expected)

    def test_update_signal_estimation(self):
        """A unit test of updating estimation by IHT or HTP.
        """
        x = np.asarray([[2.], [1], [3]])
        y = np.asarray([[5.], [7]])
        H = np.asarray([[2., 1, 0], [0, 1, 2]])
        lambda_step = 2
        iter_type = constants.HTP_NAME  # HTP
        x_updated = GradientDescent().update_signal_estimation(
            x, y, H, lambda_step, iter_type)
        x_expected = np.asarray([[5 / 2], [0], [7 / 2]])
        np.testing.assert_array_equal(x_updated, x_expected)
        x = np.asarray([[2.], [1], [3]])
        iter_type = constants.IHT_NAME  #IHT
        x_updated = GradientDescent().update_signal_estimation(
            x, y, H, lambda_step, iter_type)
        x_expected = np.asarray([[2.], [0.], [3.]])
        np.testing.assert_array_equal(x_updated, x_expected)


if __name__ == '__main__':
    unittest.main()
