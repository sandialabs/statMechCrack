"""A module for testing when something is one.

"""

import unittest
import numpy as np

from .test_zero import random_crack_model


class One(unittest.TestCase):
    """Class to test for when something is one.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_one_relative_nondimensional_reaction_rate_coefficient()

    def test_one_relative_nondimensional_reaction_rate_coefficient(self):
        """Function to test for a relative nondimensional
        reaction rate coefficient of one.

        """
        model = random_crack_model()
        for k in range(model.W):
            self.assertAlmostEqual(
                model.k_isometric(np.ones(model.W), k), 1
            )
