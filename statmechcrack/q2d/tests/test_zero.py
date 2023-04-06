"""A module for testing when something is zero.

"""

import unittest
import numpy as np

from ..core import CrackQ2D


def random_crack_model(
    L=np.random.randint(13, high=18),
    W=np.random.randint(6, high=10),
    N=None,
    kappa=88*(1 - np.random.rand()/2),
    alpha=2*(1 - np.random.rand()/2),
    varepsilon=88*(1 - np.random.rand()/2),
    periodic_boundary_conditions=False
):
    """Function to produce a random crack model.

    """
    if N is None:
        N = np.random.randint(6, high=9, size=W)
    return CrackQ2D(
        L=L, N=N, W=W,
        kappa=kappa, alpha=alpha, varepsilon=varepsilon,
        periodic_boundary_conditions=periodic_boundary_conditions
    )


class Zero(unittest.TestCase):
    """Class to test for when something is zero.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_zero_minimized_nondimensional_energy()
        self.test_zero_nondimensional_energy()
        self.test_zero_nondimensional_end_force()
        self.test_zero_nondimensional_jacobian()

    def test_zero_nondimensional_energy(self):
        """Function to test for zero nondimensional energy.

        """
        rgn = np.random.rand()
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            self.assertEqual(
                model.beta_U_0(
                    rgn*np.ones(model.W),
                    rgn*np.ones((model.L, model.W))
                ), 0
            )
            self.assertEqual(
                model.beta_U_1(
                    np.ones((model.L, model.W))
                ), 0
            )
            self.assertEqual(
                model.beta_U(
                    np.ones(model.W),
                    np.ones((model.L*model.W))
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
                    np.ones(((model.L + 1)*model.W))
                ), 0
            )

    def test_zero_nondimensional_jacobian(self):
        """Function to test for zero nondimensional Jacobian.

        """
        rgn = np.random.rand()
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = rgn*np.ones(model.W)
            s_vec = rgn*np.ones(model.L*model.W)
            j_U_0 = model.j_U_0(v, s_vec)
            for j_U_0_ik in j_U_0:
                self.assertAlmostEqual(j_U_0_ik, 0)
            s = np.random.rand(model.L, model.W)
            for j in range(model.W):
                s[-model.M[j]:, j] = 1
            j_U_1 = model.j_U_1(np.reshape(s, model.L*model.W))
            for j_U_1_ik in j_U_1:
                self.assertAlmostEqual(j_U_1_ik, 0)
            v = np.ones(model.W)
            s_vec = np.ones(model.L*model.W)
            j_U = model.j_U(v, s_vec)
            for j_U_ik in j_U:
                self.assertAlmostEqual(j_U_ik, 0)
            p = np.zeros(model.W)
            vs_vec = rgn*np.ones((model.L + 1)*model.W)
            j_Pi_0 = model.j_Pi_0(p, vs_vec)
            for j_Pi_0_ik in j_Pi_0:
                self.assertAlmostEqual(j_Pi_0_ik, 0)
            vs = np.random.rand(model.L + 1, model.W)
            for j in range(model.W):
                vs[-model.M[j]:, j] = 1
            j_Pi_1 = model.j_Pi_1(np.reshape(vs, (model.L + 1)*model.W))
            for j_Pi_1_ik in j_Pi_1:
                self.assertAlmostEqual(j_Pi_1_ik, 0)
            vs_vec = np.ones((model.L + 1)*model.W)
            j_Pi = model.j_Pi(p, vs_vec)
            for j_Pi_ik in j_Pi:
                self.assertAlmostEqual(j_Pi_ik, 0)

    def test_zero_minimized_nondimensional_energy(self):
        """Function to test for zero minimized nondimensional energy.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            self.assertAlmostEqual(
                model.minimize_beta_U(np.ones(model.W))[0], 0
            )
            self.assertAlmostEqual(
                model.minimize_beta_Pi(np.zeros(model.W))[0], 0
            )

    def test_zero_nondimensional_end_force(self):
        """Function to test for zero nondimensional end force.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            for p_k in model.p_mechanical(np.ones(model.W)):
                self.assertAlmostEqual(p_k, 0)
