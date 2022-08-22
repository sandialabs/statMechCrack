"""A module for testing boundary behavior.

"""

import unittest
import numpy as np

from .test_zero import random_crack_model


class Boundary(unittest.TestCase):
    """Class to test boundary behavior.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_boundary_nondimensional_force()
        self.test_boundary_nondimensional_end_separation()
        self.test_boundary_relative_nondimensional_gibbs_free_energy()
        self.test_boundary_relative_nondimensional_helmholtz_free_energy()
        self.test_boundary_relative_nondimensional_reaction_rate_coefficient()

    def test_boundary_nondimensional_force(self):
        """Function to test asymptotic behavior of
        the nondimensional force.

        """
        rgn = np.random.rand()
        model = random_crack_model(varepsilon=800)
        compare = model.p_0(rgn, [1, 1])[0]
        self.assertAlmostEqual(
            model.p(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )
        model = random_crack_model(N=100, varepsilon=800)
        compare = 3*model.kappa/model.N**3*(rgn - 1)
        self.assertAlmostEqual(
            model.p(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )

    def test_boundary_nondimensional_end_separation(self):
        """Function to test asymptotic behavior of
        the nondimensional end separation.

        """
        rgn = np.random.rand()
        model = random_crack_model(varepsilon=800)
        compare = model.v_0(rgn, [1, 1])[0]
        self.assertAlmostEqual(
            model.v(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )
        model = random_crack_model(N=100, varepsilon=800)
        compare = 1 + model.N**3/3/model.kappa*rgn
        self.assertAlmostEqual(
            model.v(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )

    def test_boundary_relative_nondimensional_helmholtz_free_energy(self):
        """Function to test asymptotic behavior of
        the relative nondimensional Helmholtz free energy.

        """
        rgn = np.random.rand()
        model = random_crack_model(varepsilon=800)
        compare = model.beta_A_0(rgn, [1, 1])[0]
        self.assertAlmostEqual(
            model.beta_A(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )
        model = random_crack_model(N=100, varepsilon=800)
        compare = 3*model.kappa/2/model.N**3*(rgn - 1)**2
        self.assertAlmostEqual(
            model.beta_A(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )

    def test_boundary_relative_nondimensional_gibbs_free_energy(self):
        """Function to test asymptotic behavior of
        the relative nondimensional Gibbs free energy.

        """
        rgn = np.random.rand()
        model = random_crack_model(varepsilon=800)
        compare = model.beta_G_0(rgn, [1, 1])[0]
        self.assertAlmostEqual(
            model.beta_G(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )
        model = random_crack_model(N=100, varepsilon=800)
        compare = -model.N**3/6/model.kappa*rgn**2 - rgn
        self.assertAlmostEqual(
            model.beta_G(rgn)[0], compare, delta=np.abs(1e-1*compare)
        )

    def test_boundary_relative_nondimensional_reaction_rate_coefficient(self):
        """Function to test asymptotic behavior of
        the relative nondimensional reaction rate coefficient.

        """
        rgn = np.random.rand()
        model = random_crack_model(varepsilon=800)
        compare = model.k_0(rgn, [1, 1], ensemble='isometric')[0]
        self.assertAlmostEqual(
            model.k(rgn, ensemble='isometric')[0],
            compare, delta=np.abs(1e-0*compare)
        )
        compare = model.k_0(rgn, [1, 1], ensemble='isotensional')[0]
        self.assertAlmostEqual(
            model.k(rgn, ensemble='isotensional')[0],
            compare, delta=np.abs(1e-0*compare)
        )
        model = random_crack_model(N=100, varepsilon=800)
        compare = np.exp(2*model.kappa/model.alpha/model.N**2*(rgn - 1))
        self.assertAlmostEqual(
            model.k(rgn, ensemble='isometric')[0],
            compare, delta=np.abs(1e-0*compare)
        )
        model = random_crack_model(N=100, varepsilon=800)
        rgn = 3*model.kappa/model.N**3*np.random.rand()
        compare = np.exp(2*model.N/3/model.alpha*rgn)
        self.assertAlmostEqual(
            model.k(rgn, ensemble='isotensional')[0],
            compare, delta=np.abs(1e-0*compare)
        )


if __name__ == '__main__':
    unittest.main()
