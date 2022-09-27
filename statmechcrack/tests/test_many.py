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
        self.test_many_thermodynamic_connection_isometric()
        self.test_many_thermodynamic_connection_isometric_0()
        self.test_many_thermodynamic_connection_isometric_b()
        self.test_many_thermodynamic_connection_isotensional()
        self.test_many_thermodynamic_connection_isotensional_0()
        self.test_many_thermodynamic_connection_isotensional_b()

    def test_many_inverses(self):
        """Function to test inverse calculation at many random points.

        """
        y = np.random.rand(88)
        x = BasicUtility().inv_fun(lambda x: x**2, y)
        self.assertTrue(np.allclose(y, x**2))

    def test_many_thermodynamic_connection_isometric_b(self):
        """Function to test the principal thermodynamic connection
        in the isometric ensemble for the isolated bending system
        at many random points.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        v = 1 + np.random.rand(88)
        self.assertTrue(
            np.allclose(
                model.Q_b_isometric(v, [rgn0, rgn1]),
                np.exp(-model.beta_A_b(
                    v, [rgn0, rgn1], ensemble='isometric', absolute=True
                ))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(1/model.Q_b_isometric(v, [rgn0, rgn1])),
                model.beta_A_b(
                    v, [rgn0, rgn1], ensemble='isometric', absolute=True
                )
            )
        )

    def test_many_thermodynamic_connection_isometric_0(self):
        """Function to test the principal thermodynamic connection
        in the isometric ensemble for the reference system
        at many random points.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        v = 1 + np.random.rand(88)
        self.assertTrue(
            np.allclose(
                model.Q_0_isometric(v, [rgn0, rgn1]),
                np.exp(-model.beta_A_0(
                    v, [rgn0, rgn1], ensemble='isometric', absolute=True
                ))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(1/model.Q_0_isometric(v, [rgn0, rgn1])),
                model.beta_A_0(
                    v, [rgn0, rgn1], ensemble='isometric', absolute=True
                )
            )
        )

    def test_many_thermodynamic_connection_isometric(self):
        """Function to test the principal thermodynamic connection
        in the isometric ensemble for the asymptotic appoximation
        of the full system at many random points.

        """
        model = random_crack_model()
        v = 1 + np.random.rand(88)
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
                np.exp(
                    -model.beta_A_abs_isometric(v, transition_state=True)
                )
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(model.Q_isometric(v, transition_state=True)),
                -model.beta_A_abs_isometric(v, transition_state=True)
            )
        )

    def test_many_thermodynamic_connection_isotensional_b(self):
        """Function to test the principal thermodynamic connection
        in the isotensional ensemble for the isolated bending system
        at many random points.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        v = 1 + np.random.rand(88)
        p = model.p(v, ensemble='isotensional')
        self.assertTrue(
            np.allclose(
                model.Z_b_isotensional(p, [rgn0, rgn1]),
                np.exp(-model.beta_G_b(
                    p, [rgn0, rgn1], ensemble='isotensional', absolute=True
                ))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(1/model.Z_b_isotensional(p, [rgn0, rgn1])),
                model.beta_G_b(
                    p, [rgn0, rgn1], ensemble='isotensional', absolute=True
                )
            )
        )

    def test_many_thermodynamic_connection_isotensional_0(self):
        """Function to test the principal thermodynamic connection
        in the isotensional ensemble for the reference system
        at many random points.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        v = 1 + np.random.rand(88)
        p = model.p(v, ensemble='isotensional')
        self.assertTrue(
            np.allclose(
                model.Z_0_isotensional(p, [rgn0, rgn1]),
                np.exp(-model.beta_G_0(
                    p, [rgn0, rgn1], ensemble='isotensional', absolute=True
                ))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(1/model.Z_0_isotensional(p, [rgn0, rgn1])),
                model.beta_G_0(
                    p, [rgn0, rgn1], ensemble='isotensional', absolute=True
                )
            )
        )

    def test_many_thermodynamic_connection_isotensional(self):
        """Function to test the principal thermodynamic connection
        in the isotensional ensemble for the asymptotic appoximation
        of the full system at many random points.

        """
        model = random_crack_model()
        v = 1 + np.random.rand(88)
        p = model.p(v, ensemble='isotensional')
        self.assertTrue(
            np.allclose(
                model.Z_isotensional(p),
                np.exp(-model.beta_G(
                    p, ensemble='isotensional', absolute=True
                ))
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(model.Z_isotensional(p)),
                -model.beta_G(
                    p, ensemble='isotensional', absolute=True
                )
            )
        )
        model = random_crack_model()
        self.assertTrue(
            np.allclose(
                model.Z_isotensional(p, transition_state=True),
                np.exp(
                    -model.beta_G_abs_isotensional(p, transition_state=True)
                )
            )
        )
        self.assertTrue(
            np.allclose(
                np.log(model.Z_isotensional(p, transition_state=True)),
                -model.beta_G_abs_isotensional(p, transition_state=True)
            )
        )


if __name__ == '__main__':
    unittest.main()
