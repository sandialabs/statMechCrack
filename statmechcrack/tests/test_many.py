"""A module for testing many or more complicated things.

"""

import unittest
import numpy as np

from ..utility import BasicUtility
from .test_zero import random_crack_model


class Many(unittest.TestCase):
    """Many (or More complex)

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_many_inverses()
        self.test_many_isometric_Q_beta_A()

    def test_many_inverses(self):
        """Function to test inverse calculation at many random points.

        """
        y = np.random.rand(88)
        x = BasicUtility().inv_fun(lambda x: x**2, y)
        self.assertTrue(np.isclose(y, x**2).all())

    def test_many_isometric_Q_b_beta_A_b(self):
        """Function to test the principal thermodynamic connection
        in the isometric ensemble for the isolated bending system
        at many random points.

        """
        v = 10*np.random.rand(88)
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        self.assertTrue((
            model.Q_b_isometric(v, [rgn0, rgn1]) ==
            np.exp(-model.beta_A_b(
                v, [rgn0, rgn1], ensemble='isometric', absolute=True
            ))
        ).all())
        self.assertTrue((
            np.log(1/model.Q_b_isometric(v, [rgn0, rgn1])) ==
            model.beta_A_b(
                v, [rgn0, rgn1], ensemble='isometric', absolute=True
            )
        ).all())

    def test_many_isometric_Q_0_beta_A_0(self):
        """Function to test the principal thermodynamic connection
        in the isometric ensemble for the reference system
        at many random points.

        """
        v = 10*np.random.rand(88)
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        self.assertTrue((
            model.Q_0_isometric(v, [rgn0, rgn1]) ==
            np.exp(-model.beta_A_0(
                v, [rgn0, rgn1], ensemble='isometric', absolute=True
            ))
        ).all())
        self.assertTrue((
            np.log(1/model.Q_0_isometric(v, [rgn0, rgn1])) ==
            model.beta_A_0(
                v, [rgn0, rgn1], ensemble='isometric', absolute=True
            )
        ).all())

    def test_many_isometric_Q_beta_A(self):
        """Function to test the principal thermodynamic connection
        in the isometric ensemble for the asymptotic appoximation
        of the full system at many random points.

        """
        v = 10*np.random.rand(88)
        model = random_crack_model()
        self.assertTrue(
            np.allclose(
                model.Q_isometric(v),
                np.exp(-model.beta_A(
                    v, ensemble='isometric', absolute=True
                ))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(model.Q_isometric(v)),
                -model.beta_A(
                    v, ensemble='isometric', absolute=True
                )
            )
        )
        model = random_crack_model()
        self.assertTrue(
            np.allclose(
                model.Q_isometric(v, transition_state=True),
                np.exp(-model.beta_A_abs_isometric(v, transition_state=True))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(model.Q_isometric(v, transition_state=True)),
                -model.beta_A_abs_isometric(v, transition_state=True)
            )
        )


if __name__ == '__main__':
    unittest.main()
