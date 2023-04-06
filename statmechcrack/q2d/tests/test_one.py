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
        self.test_one_nondimensional_relative_reaction_rate_coefficient()

    def test_one_nondimensional_relative_reaction_rate_coefficient(self):
        """Function to test for a relative nondimensional
        reaction rate coefficient of one.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            for k in range(model.W):
                self.assertAlmostEqual(
                    model.k_isometric(np.ones(model.W), k), 1
                )
