"""

"""

import unittest
import numpy as np

from ..utility import BasicUtility


class Many(unittest.TestCase):
    """Many (or More complex)

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_many_inverses()

    def test_many_inverses(self):
        """Function to test inverse calculation at many random points.

        """
        y = np.random.rand(88)
        x = BasicUtility().inv_fun(lambda x: x**2, y)
        self.assertTrue(np.isclose(y, x**2).all())


if __name__ == '__main__':
    unittest.main()
