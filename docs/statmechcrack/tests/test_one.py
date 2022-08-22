"""A module for testing when something is one.

"""

import unittest
import numpy as np

from ..utility import BasicUtility
from .test_zero import random_crack_model


class One(unittest.TestCase):
    """Class to test for when something is one.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_one_inverse()
        self.test_one_nondimensional_end_separation()
        self.test_one_relative_nondimensional_reaction_rate_coefficient()

    def test_one_inverse(self):
        """Function to test inverse calculation at one.

        """
        self.assertEqual(BasicUtility().inv_fun(lambda x: x, 1), 1)

    def test_one_nondimensional_end_separation(self):
        """Function to test for a nondimensional end separation of one.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        self.assertAlmostEqual(
            model.v_b_mechanical(
                0, [rgn, rgn]
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_0_mechanical(
                0, [rgn, rgn]
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_mechanical(
                0
            )[0], 1
        )
        self.assertAlmostEqual(
            model.v_b_isotensional(
                0, [rgn, rgn]
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_0_isotensional(
                0, [rgn, rgn]
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_isotensional(
                0, approach='asymptotic'
            )[0], 1
        )
        self.assertAlmostEqual(
            model.v_b(
                0, [rgn, rgn], ensemble='isometric'
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_b(
                0, [rgn, rgn], ensemble='isotensional'
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_0(
                0, [rgn, rgn], ensemble='isometric'
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v_0(
                0, [rgn, rgn], ensemble='isotensional'
            )[0], rgn
        )
        self.assertAlmostEqual(
            model.v(
                0, ensemble='isometric', approach='asymptotic'
            )[0], 1
        )
        self.assertAlmostEqual(
            model.v(
                0, ensemble='isotensional', approach='asymptotic'
            )[0], 1
        )

    def test_one_relative_nondimensional_reaction_rate_coefficient(self):
        """Function to test for a relative nondimensional
        reaction rate coefficient of one.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        self.assertEqual(
            model.k_b_isometric(
                1, [rgn0, rgn1]
            )[0], 1
        )
        self.assertEqual(
            model.k_0_isometric(
                1, [rgn0, rgn1]
            )[0], 1
        )
        self.assertEqual(
            model.k_isometric(
                1, approach='asymptotic'
            )[0], 1
        )
        self.assertEqual(
            model.k_b_isotensional(
                0, [rgn0, rgn1]
            )[0], 1
        )
        self.assertEqual(
            model.k_0_isotensional(
                0, [rgn0, rgn1]
            )[0], 1
        )
        self.assertEqual(
            model.k_isotensional(
                0, approach='asymptotic'
            )[0], 1
        )
        self.assertEqual(
            model.k_b(
                1, [rgn0, rgn1], ensemble='isometric'
            )[0], 1
        )
        self.assertEqual(
            model.k_0(
                1, [rgn0, rgn1], ensemble='isometric'
            )[0], 1
        )
        self.assertEqual(
            model.k(
                1, ensemble='isometric', approach='asymptotic'
            )[0], 1
        )
        self.assertEqual(
            model.k_b(
                0, [rgn0, rgn1], ensemble='isotensional'
            )[0], 1
        )
        self.assertEqual(
            model.k_0(
                0, [rgn0, rgn1], ensemble='isotensional'
            )[0], 1
        )
        self.assertEqual(
            model.k(
                0, ensemble='isotensional', approach='asymptotic'
            )[0], 1
        )


if __name__ == '__main__':
    unittest.main()
