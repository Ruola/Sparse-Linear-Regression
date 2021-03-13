import numpy as np
import unittest

from change_condition_number import ChangeConditionNumber
import utils.constants as constants


class TestChangeConditionNumber(unittest.TestCase):
    """A unit test for ChangeConditionNumber.
    """
    def test_update_algos_map(self):
        """A unit test of ChangeConditionNumber()._update_algos_map().
        """
        algo_name, kappa, error = "IHT", 3.1, 33.5
        map = ChangeConditionNumber()._update_algos_map(dict(), algo_name, kappa, error)
        expected_map = {algo_name:{kappa: error}}
        self.assertEquals(expected_map, map)

if __name__ == '__main__':
    unittest.main()
