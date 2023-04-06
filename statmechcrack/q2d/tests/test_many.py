"""A module for testing many or more complicated things.

"""

import unittest
import numpy as np
from time import time
from copy import deepcopy
from scipy.optimize import minimize

from .test_zero import random_crack_model


h = 1e-5


class Many(unittest.TestCase):
    """Many (or More complex)

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_nondimensional_bending_energy_calculation()
        self.test_nondimensional_energy_minimization_no_hessian()
        self.test_nondimensional_energy_minimization_no_hessian_ts()
        self.test_nondimensional_energy_minimization_no_jacobian_hessian()
        self.test_nondimensional_energy_minimization_no_jacobian_hessian_ts()
        self.test_nondimensional_mechanical_forces()
        self.test_nondimensional_relative_reaction_rate_coefficients_symmetry()

    def test_nondimensional_relative_reaction_rate_coefficients_symmetry(self):
        """Function to test that the
        nondimensional relative reaction rate coefficients are equal
        for a flat crack front and peridic boundary conditions
        when the loads are also equal.

        """
        L = np.random.randint(13, high=18)
        W = np.random.randint(6, high=10)
        model = random_crack_model(
            L=L, W=W,
            N=np.array(
                np.random.randint(6, high=9)*np.ones(W),
                dtype=int
            ),
            periodic_boundary_conditions=True
        )
        v = np.random.rand()*np.ones(model.W)
        rate_ref = model.k_isometric(v, 0)
        for k in range(1, model.W):
            self.assertAlmostEqual(rate_ref, model.k_isometric(v, k), places=6)

    def test_nondimensional_bending_energy_calculation(self):
        """Function to test the nondimensional bending energy calculation.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = np.random.rand(model.W)
            self.assertEqual(v.shape, (model.W,))
            s = np.random.rand(model.L, model.W)
            self.assertEqual(s.shape, (model.L, model.W))
            beta_U_0_model = model.beta_U_0(v, s)
            vs = np.concatenate(([v], s))
            self.assertEqual(vs.shape, (model.L + 1, model.W))
            beta_U_0_check = 0
            for i in range(2, model.L + 1):
                for k in range(model.W):
                    beta_U_0_check += model.kappa/2*(
                        (vs[i - 2, k] - 2*vs[i - 1, k] + vs[i, k])
                    )**2
            for i in range(model.L + 1):
                for k in range(2, model.W):
                    beta_U_0_check += model.kappa/2*(
                        (vs[i, k - 2] - 2*vs[i, k - 1] + vs[i, k])
                    )**2
                if model.periodic_boundary_conditions is True:
                    beta_U_0_check += model.kappa/2*(
                        (vs[i, model.W - 2] - 2*vs[i, model.W - 1] + vs[i, 0])
                    )**2 + model.kappa/2*(
                        (vs[i, model.W - 1] - 2*vs[i, 0] + vs[i, 1])
                    )**2
            self.assertAlmostEqual(beta_U_0_model, beta_U_0_check)

    def test_nondimensional_mechanical_forces(self):
        """Function to test the nondimensional bending energy calculation.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = np.random.rand(model.W)
            p = model.p_mechanical(v)
            _, p_same, s, _ = model.minimize_beta_U(v)
            for k, p_k in enumerate(p):
                self.assertAlmostEqual(p_k, p_same[k])
                v_h_p = deepcopy(v)
                v_h_m = deepcopy(v)
                v_h_p[k] = v_h_p[k] + h/2
                v_h_m[k] = v_h_m[k] - h/2
                p_check_k = (model.beta_U(v_h_p, s) - model.beta_U(v_h_m, s))/h
                self.assertAlmostEqual(p_k, p_check_k, delta=h)

    def test_nondimensional_energy_minimization_no_hessian(self):
        """Function to test the nondimensional energy minimization
        is the same without the hessian.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = np.random.rand(model.W)
            time_0 = time()
            beta_U, _, s, _ = model.minimize_beta_U(v)
            time_1 = time()
            s_vec_guess = np.ones(model.L*model.W)
            res = minimize(
                lambda s_vec: model.beta_U(v, s_vec),
                s_vec_guess,
                method='Newton-CG',
                jac=lambda s_vec: model.j_U(v, s_vec)
            )
            beta_U_check = res.fun
            s_check = np.resize(res.x, (model.L, model.W))
            time_2 = time()
            self.assertGreater(time_2 - time_1, time_1 - time_0)
            self.assertTrue(
                np.abs((beta_U - beta_U_check)/beta_U_check) < 1e-3
            )
            self.assertTrue(
                (np.abs((s - s_check)/s_check) < 5e-2).all()
            )

    def test_nondimensional_energy_minimization_no_hessian_ts(self):
        """Function to test the nondimensional energy minimization
        in a transition state is the same without the hessian.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = np.random.rand(model.W)
            transition_state = np.random.randint(0, high=model.W)
            time_0 = time()
            beta_U, _, s, _ = model.minimize_beta_U(
                v,
                transition_state=transition_state
            )
            self.assertEqual(
                s[model.N[transition_state], transition_state],
                model.lambda_TS
            )
            time_1 = time()
            s_vec_guess = np.ones(
                model.L*model.W - (transition_state is not None)
            )
            res = minimize(
                lambda s_vec: model.beta_U(
                    v, s_vec, transition_state=transition_state
                ),
                s_vec_guess,
                method='Newton-CG',
                jac=lambda s_vec: model.j_U(
                    v, s_vec, transition_state=transition_state
                )
            )
            beta_U_check = res.fun
            s_vec_check = res.x
            s_vec_check = np.insert(
                s_vec_check,
                np.ravel_multi_index(
                    (model.N[transition_state], transition_state),
                    (model.L, model.W)
                ),
                model.lambda_TS
            )
            s_check = np.resize(s_vec_check, (model.L, model.W))
            time_2 = time()
            self.assertGreater(time_2 - time_1, time_1 - time_0)
            self.assertTrue(
                np.abs((beta_U - beta_U_check)/beta_U_check) < 1e-3
            )
            self.assertTrue(
                (np.abs((s - s_check)/s_check) < 5e-2).all()
            )

    def test_nondimensional_energy_minimization_no_jacobian_hessian(self):
        """Function to test the nondimensional energy minimization
        is the same without the jacobian or hessian.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = np.random.rand(model.W)
            time_0 = time()
            beta_U, _, s, _ = model.minimize_beta_U(v)
            time_1 = time()
            s_vec_guess = np.ones(model.L*model.W)
            res = minimize(
                lambda s_vec: model.beta_U(v, s_vec),
                s_vec_guess
            )
            beta_U_check = res.fun
            s_check = np.resize(res.x, (model.L, model.W))
            time_2 = time()
            self.assertGreater(time_2 - time_1, time_1 - time_0)
            self.assertTrue(
                np.abs((beta_U - beta_U_check)/beta_U_check) < 1e-3
            )
            self.assertTrue(
                (np.abs((s - s_check)/s_check) < 5e-2).all()
            )

    def test_nondimensional_energy_minimization_no_jacobian_hessian_ts(self):
        """Function to test the nondimensional energy minimization
        in a transition state is the same without the hessian.

        """
        models = (
            random_crack_model(),
            random_crack_model(periodic_boundary_conditions=True)
        )
        for model in models:
            v = np.random.rand(model.W)
            transition_state = np.random.randint(0, high=model.W)
            time_0 = time()
            beta_U, _, s, _ = model.minimize_beta_U(
                v,
                transition_state=transition_state
            )
            self.assertEqual(
                s[model.N[transition_state], transition_state],
                model.lambda_TS
            )
            time_1 = time()
            s_vec_guess = np.ones(
                model.L*model.W - (transition_state is not None)
            )
            res = minimize(
                lambda s_vec: model.beta_U(
                    v, s_vec, transition_state=transition_state
                ),
                s_vec_guess
            )
            beta_U_check = res.fun
            s_vec_check = res.x
            s_vec_check = np.insert(
                s_vec_check,
                np.ravel_multi_index(
                    (model.N[transition_state], transition_state),
                    (model.L, model.W)
                ),
                model.lambda_TS
            )
            s_check = np.resize(s_vec_check, (model.L, model.W))
            time_2 = time()
            self.assertGreater(time_2 - time_1, time_1 - time_0)
            self.assertTrue(
                np.abs((beta_U - beta_U_check)/beta_U_check) < 1e-3
            )
            self.assertTrue(
                (np.abs((s - s_check)/s_check) < 5e-2).all()
            )


if __name__ == '__main__':
    unittest.main()
