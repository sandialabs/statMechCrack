"""A module for testing when something is zero.

"""

import unittest
import numpy as np

from ..core import Crack
from ..utility import BasicUtility


def random_crack_model(
    N=np.random.randint(5, high=23),
    M=np.random.randint(5, high=23),
    kappa=88*(1 - np.random.rand()/2),
    alpha=2*(1 - np.random.rand()/2),
    varepsilon=88*(1 - np.random.rand()/2)
):
    """Function to produce a random crack model.

    Args:
        N (int, optional, default=random): The number of broken bonds.
        M (int, optional, default=random): The number of intact bonds.
        kappa (float, optional, default=random):
            The nondimensional bending stiffness.
        alpha (float, optional, default=random):
            The nondimensional Morse parameter.
        varepsilon (float, optional, default=random):
            The nondimensional bond energy.

    Returns:
        object: A crack model instance.

    """
    return Crack(N=N, M=M, kappa=kappa, alpha=alpha, varepsilon=varepsilon)


class Zero(unittest.TestCase):
    """Class to test for when something is zero.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_zero_inverse()
        self.test_zero_nondimensional_energy()
        self.test_zero_nondimensional_jacobian()
        self.test_zero_nondimensional_end_force()
        self.test_zero_minimized_nondimensional_energy()
        self.test_zero_relative_nondimensional_gibbs_free_energy()
        self.test_zero_relative_nondimensional_helmholtz_free_energy()

    def test_zero_inverse(self):
        """Function to test inverse calculation at zero.

        """
        self.assertEqual(BasicUtility().inv_fun(lambda x: x, 0), 0)

    def test_zero_nondimensional_end_force(self):
        """Function to test for zero nondimensional end force.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        self.assertAlmostEqual(
            model.p_b_mechanical(rgn, [rgn, rgn])[0], 0
        )
        self.assertAlmostEqual(
            model.p_0_mechanical(rgn, [rgn, rgn])[0], 0
        )
        self.assertAlmostEqual(
            model.p_mechanical(1)[0], 0
        )
        self.assertAlmostEqual(
            model.p_b_isometric(rgn, [rgn, rgn])[0], 0
        )
        self.assertAlmostEqual(
            model.p_0_isometric(rgn, [rgn, rgn])[0], 0
        )
        self.assertAlmostEqual(
            model.p_isometric(1, approach='asymptotic')[0], 0
        )
        self.assertAlmostEqual(
            model.p_b(rgn, [rgn, rgn], ensemble='isometric')[0], 0
        )
        self.assertAlmostEqual(
            model.p_b(rgn, [rgn, rgn], ensemble='isotensional')[0], 0
        )
        self.assertAlmostEqual(
            model.p_0(rgn, [rgn, rgn], ensemble='isometric')[0], 0
        )
        self.assertAlmostEqual(
            model.p_0(rgn, [rgn, rgn], ensemble='isotensional')[0], 0
        )
        self.assertAlmostEqual(
            model.p(1, ensemble='isometric', approach='asymptotic')[0], 0
        )
        self.assertAlmostEqual(
            model.p(1, ensemble='isotensional', approach='asymptotic')[0], 0
        )

    def test_zero_nondimensional_energy(self):
        """Function to test for zero nondimensional energy.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        self.assertEqual(model.beta_U_00(rgn, rgn*np.ones(8)), 0)
        self.assertEqual(model.beta_U_01(rgn*np.ones(8)), 0)
        self.assertEqual(model.beta_U_0(rgn, rgn*np.ones(8)), 0)
        self.assertEqual(model.beta_U_1(np.ones(8)), 0)
        self.assertEqual(model.beta_U(1, np.ones(8)), 0)
        self.assertEqual(model.beta_Pi_00(0, rgn, rgn*np.ones(16)), 0)
        self.assertEqual(model.beta_Pi_0(0, rgn, rgn*np.ones(16)), 0)
        self.assertEqual(model.beta_Pi(0, 1, np.ones(16)), 0)

    def test_zero_nondimensional_jacobian(self):
        """Function to test for zero nondimensional Jacobian.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        for j_i in model.j_U_00(rgn, rgn*np.ones(model.L)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_U_0(rgn, rgn*np.ones(model.L)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_U_1(np.ones(model.M)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_U(1, np.ones(model.L)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_Pi_00(0, rgn, rgn*np.ones(model.L)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_Pi_0(0, rgn, rgn*np.ones(model.L)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_Pi_1(np.ones(model.M)):
            self.assertAlmostEqual(j_i, 0)
        for j_i in model.j_Pi(0, 1, np.ones(model.L)):
            self.assertAlmostEqual(j_i, 0)

    def test_zero_minimized_nondimensional_energy(self):
        """Function to test for zero minimized nondimensional energy.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        self.assertAlmostEqual(
            model.minimize_beta_U_00(rgn0, [rgn0, rgn0])[0][0], 0
        )
        self.assertAlmostEqual(
            model.minimize_beta_U(1)[0][0], 0
        )
        self.assertAlmostEqual(
            model.minimize_beta_Pi_00(0, [rgn0, rgn1])[0][0], 0
        )
        self.assertAlmostEqual(
            model.minimize_beta_Pi(0)[0][0], 0
        )

    def test_zero_relative_nondimensional_helmholtz_free_energy(self):
        """Function to test for zero relative Helmholtz free energy.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        self.assertEqual(
            model.beta_A_b_isometric(
                1, [rgn, rgn]
            ), 0
        )
        self.assertEqual(
            model.beta_A_0_isometric(
                1, [rgn, rgn]
            ), 0
        )
        self.assertEqual(
            model.beta_A_isometric(
                1, approach='asymptotic'
            ), 0
        )
        self.assertEqual(
            model.beta_A_b(
                1, [rgn, rgn], ensemble='isometric'
            )[0], 0
        )
        self.assertAlmostEqual(
            model.beta_A_b(
                rgn, [rgn, rgn], ensemble='isotensional'
            )[0], 0
        )
        self.assertEqual(
            model.beta_A_0(
                1, [rgn, rgn], ensemble='isometric'
            )[0], 0
        )
        self.assertAlmostEqual(
            model.beta_A_0(
                rgn, [rgn, rgn], ensemble='isotensional'
            )[0], 0
        )
        self.assertEqual(
            model.beta_A(
                1, ensemble='isometric', approach='asymptotic'
            )[0], 0
        )
        self.assertAlmostEqual(
            model.beta_A(
                1, ensemble='isotensional', approach='asymptotic'
            )[0], 0
        )
        self.assertEqual(
            model.beta_A(
                1, ensemble='isometric', approach='monte carlo',
                num_processes=2, num_burns=88, num_samples=88
            )[0], 0
        )
        self.assertEqual(
            model.beta_A_isometric(
                1, approach='monte carlo',
                num_processes=2, num_burns=88, num_samples=88
            )[0], 0
        )
        self.assertEqual(
            model.beta_A_isometric_monte_carlo(
                1, num_processes=2, num_burns=88, num_samples=88
            )[0], 0
        )

    def test_zero_relative_nondimensional_gibbs_free_energy(self):
        """Function to test for zero relative Gibbs free energy.

        """
        rgn0, rgn1 = np.random.rand(2)
        model = random_crack_model()
        self.assertEqual(
            model.beta_G_b_isotensional(
                0, [rgn0, rgn1]
            ), 0
        )
        self.assertEqual(
            model.beta_G_0_isotensional(
                0, [rgn0, rgn1]
            ), 0
        )
        self.assertEqual(
            model.beta_G_isotensional(
                0, approach='asymptotic'
            ), 0
        )
        self.assertAlmostEqual(
            model.beta_G_b(
                0, [1, 1], ensemble='isometric'
            )[0], 0
        )
        self.assertEqual(
            model.beta_G_b(
                0, [rgn0, rgn1], ensemble='isotensional'
            )[0], 0
        )
        self.assertAlmostEqual(
            model.beta_G_0(
                0, [1, 1], ensemble='isometric'
            )[0], 0
        )
        self.assertEqual(
            model.beta_G_0(
                0, [rgn0, rgn1], ensemble='isotensional'
            )[0], 0
        )
        self.assertAlmostEqual(
            model.beta_G(
                0, ensemble='isometric', approach='asymptotic'
            )[0], 0
        )
        self.assertEqual(
            model.beta_G(
                0, ensemble='isotensional', approach='asymptotic'
            )[0], 0
        )
        self.assertEqual(
            model.beta_G(
                0, ensemble='isotensional', approach='monte carlo',
                num_processes=2, num_burns=88, num_samples=88
            )[0], 0
        )
        self.assertEqual(
            model.beta_G_isotensional(
                0, approach='monte carlo',
                num_processes=2, num_burns=88, num_samples=88
            )[0], 0
        )
        self.assertEqual(
            model.beta_G_isotensional_monte_carlo(
                0, num_processes=2, num_burns=88, num_samples=88
            )[0], 0
        )


if __name__ == '__main__':
    unittest.main()
