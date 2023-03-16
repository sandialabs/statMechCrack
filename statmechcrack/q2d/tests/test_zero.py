"""A module for testing when something is zero.

"""

import unittest
import numpy as np

from ..core import CrackQ2D


def random_crack_model(
    N=np.random.randint(5, high=8),
    M=np.random.randint(5, high=8),
    W=np.random.randint(5, high=8),
    kappa=88*(1 - np.random.rand()/2),
    alpha=2*(1 - np.random.rand()/2),
    varepsilon=88*(1 - np.random.rand()/2)
):
    """Function to produce a random crack model.

    """
    return CrackQ2D(
        N=N, M=M, W=W, kappa=kappa, alpha=alpha, varepsilon=varepsilon
    )


class Zero(unittest.TestCase):
    """Class to test for when something is zero.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_zero_nondimensional_energy()
        self.test_zero_nondimensional_jacobian()
        self.test_zero_nondimensional_end_force()
        self.test_zero_minimized_nondimensional_energy()

    def test_zero_nondimensional_energy(self):
        """Function to test for zero nondimensional energy.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        self.assertEqual(
            model.beta_U_0(
                rgn*np.ones(model.W),
                rgn*np.ones((model.L, model.W))
            ), 0
        )
        self.assertEqual(
            model.beta_U_1(
                np.ones((model.M, model.W))
            ), 0
        )
        self.assertEqual(
            model.beta_U(
                np.ones(model.W),
                np.ones((model.L, model.W))
            ), 0
        )
        self.assertEqual(
            model.beta_Pi_0(
                np.zeros(model.W),
                rgn*np.ones(model.W),
                rgn*np.ones((model.L, model.W))
            ), 0
        )
        self.assertEqual(
            model.beta_Pi(
                np.zeros(model.W),
                np.ones(model.W),
                np.ones((model.L, model.W))
            ), 0
        )

    def test_zero_nondimensional_jacobian(self):
        """Function to test for zero nondimensional Jacobian.

        """
        rgn = np.random.rand()
        model = random_crack_model()
        v = rgn*np.ones(model.W)
        s_vec = rgn*np.ones(model.L*model.W)
        j_U_0 = model.j_U_0(v, s_vec)
        for j_U_0_ik in j_U_0:
            self.assertAlmostEqual(j_U_0_ik, 0)
        s = np.random.rand(model.L, model.W)
        s[-model.M:, :] = 1
        j_U_1 = model.j_U_1(np.reshape(s, (model.L, model.W)))
        for j_U_1_ik in j_U_1:
            self.assertAlmostEqual(j_U_1_ik, 0)
        v = np.ones(model.W)
        s_vec = np.ones(model.L*model.W)
        j_U = model.j_U(v, s_vec)
        for j_U_ik in j_U:
            self.assertAlmostEqual(j_U_ik, 0)

    def test_zero_minimized_nondimensional_energy(self):
        """Function to test for zero minimized nondimensional energy.

        """
        model = random_crack_model()
        self.assertAlmostEqual(
            model.minimize_beta_U(np.ones(model.W))[0], 0
        )

    def test_zero_nondimensional_end_force(self):
        """Function to test for zero nondimensional end force.

        """
        model = random_crack_model()
        for p_k in model.p_mechanical(np.ones(model.W)):
            self.assertAlmostEqual(p_k, 0)
