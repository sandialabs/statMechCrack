"""A module for testing many or more complicated things.

"""

import unittest
import numpy as np
from copy import deepcopy

from .test_zero import random_crack_model


h = 1e-5


class Many(unittest.TestCase):
    """Many (or More complex)

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_nondimensional_bending_energy_calculation()
        self.test_nondimensional_mechanical_forces()

    def test_nondimensional_bending_energy_calculation(self):
        """Function to test the nondimensional bending energy calculation.

        """
        model = random_crack_model()
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
        self.assertAlmostEqual(beta_U_0_model, beta_U_0_check)

    def test_nondimensional_mechanical_forces(self):
        """Function to test the nondimensional bending energy calculation.

        """
        model = random_crack_model()
        v = np.random.rand(model.W)
        p = model.p_mechanical(v)
        _, p_same, s = model.minimize_beta_U(v)
        for k, p_k in enumerate(p):
            self.assertAlmostEqual(p_k, p_same[k])
            v_h_p = deepcopy(v)
            v_h_m = deepcopy(v)
            v_h_p[k] = v_h_p[k] + h/2
            v_h_m[k] = v_h_m[k] - h/2
            p_check_k = (model.beta_U(v_h_p, s) - model.beta_U(v_h_m, s))/h
            self.assertAlmostEqual(p_k, p_check_k, delta=h)


if __name__ == '__main__':
    unittest.main()
