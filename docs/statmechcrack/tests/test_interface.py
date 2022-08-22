"""A module for testing interfaces.

"""

import unittest
import numpy as np

from ..utility import BasicUtility


class Interface(unittest.TestCase):
    """Class to test interfaces.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        pass
        self.test_interfaces_inverse()

    def test_interfaces_inverse(self):
        """Function to test inverse interface.

        """
        self.assertEqual(BasicUtility().inv_fun(lambda x: x, 0).shape, (1,))
        y = np.random.rand(8)
        self.assertEqual(BasicUtility().inv_fun(lambda x: x, y).shape, y.shape)


if __name__ == '__main__':
    unittest.main()
