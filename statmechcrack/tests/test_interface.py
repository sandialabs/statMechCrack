"""A module for testing interfaces.

"""

import unittest
import numpy as np

from ..utility import BasicUtility
from .test_zero import random_crack_model


class Interface(unittest.TestCase):
    """Class to test interfaces.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_interface_inverse()
        self.test_interface_nondimensional_end_force()
        self.test_interface_nondimensional_end_separation()
        self.test_interface_relative_nondimensional_gibbs_free_energy()
        self.test_interface_relative_nondimensional_helmholtz_free_energy()
        self.test_interface_relative_nondimensional_reaction_rate_coefficient()

    def test_interface_inverse(self):
        """Function to test inverse interface.

        """
        self.assertEqual(BasicUtility().inv_fun(lambda x: x, 0).shape, (1,))
        y = np.random.rand(8)
        self.assertEqual(BasicUtility().inv_fun(lambda x: x, y).shape, y.shape)

    def test_interface_nondimensional_force(self):
        """Function to test nondimensional end force interface.

        """
        p = random_crack_model().p
        for v in 1 + 1e-1*np.random.rand(), 1 + 1e-1*np.random.rand(8):
            _ = p(v)
            _ = p(v, ensemble='isometric')
            _ = p(v, ensemble='isotensional')
            _ = p(v, approach='asymptotic')
            _ = p(v, ensemble='isometric', approach='asymptotic')
            _ = p(v, ensemble='isotensional', approach='asymptotic')
            _ = p(v, approach='monte carlo',
                  num_processes=1, num_burns=8, num_samples=8)
            _ = p(v, approach='monte carlo', ensemble='isometric',
                  num_processes=1, num_burns=8, num_samples=8)
            _ = p(v, approach='monte carlo', ensemble='isotensional',
                  num_processes=1, num_burns=8, num_samples=8)

    def test_interface_nondimensional_end_separation(self):
        """Function to test nondimensional end separation interface.

        """
        v = random_crack_model().v
        for p in 1e-1*np.random.rand(), 1e-1*np.random.rand(8):
            _ = v(p)
            _ = v(p, ensemble='isometric')
            _ = v(p, ensemble='isotensional')
            _ = v(p, approach='asymptotic')
            _ = v(p, ensemble='isometric', approach='asymptotic')
            _ = v(p, ensemble='isotensional', approach='asymptotic')
            _ = v(p, approach='monte carlo',
                  num_processes=1, num_burns=8, num_samples=8)
            _ = v(p, approach='monte carlo', ensemble='isometric',
                  num_processes=1, num_burns=8, num_samples=8)
            _ = v(p, approach='monte carlo', ensemble='isotensional',
                  num_processes=1, num_burns=8, num_samples=8)

    def test_interface_relative_nondimensional_helmholtz_free_energy(self):
        """Function to test relative Helmholtz free energy interface.

        """
        beta_A = random_crack_model().beta_A
        for v in 1 + 1e-1*np.random.rand(), 1 + 1e-1*np.random.rand(8):
            _ = beta_A(v)
            _ = beta_A(v, ensemble='isometric')
            _ = beta_A(v, ensemble='isotensional')
            _ = beta_A(v, approach='asymptotic')
            _ = beta_A(v, absolute=False)
            _ = beta_A(v, absolute=True)
            _ = beta_A(v, ensemble='isometric', approach='asymptotic')
            _ = beta_A(v, ensemble='isotensional', approach='asymptotic')
            _ = beta_A(v, ensemble='isometric', absolute=False)
            _ = beta_A(v, ensemble='isotensional', absolute=False)
            _ = beta_A(v, ensemble='isometric', absolute=True)
            _ = beta_A(v, ensemble='isotensional', absolute=True)
            _ = beta_A(v, absolute=False, approach='asymptotic')
            _ = beta_A(v, absolute=True, approach='asymptotic')
            _ = beta_A(v, ensemble='isometric',
                       approach='asymptotic', absolute=False)
            _ = beta_A(v, ensemble='isometric',
                       approach='asymptotic', absolute=False)
            _ = beta_A(v, ensemble='isotensional',
                       approach='asymptotic', absolute=True)
            _ = beta_A(v, ensemble='isotensional',
                       approach='asymptotic', absolute=True)
            _ = beta_A(v, approach='monte carlo',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo', ensemble='isometric',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo', ensemble='isotensional',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo',
                       absolute=False,
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo',
                       absolute=False, ensemble='isometric',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo',
                       absolute=False, ensemble='isotensional',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo',
                       absolute=True,
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo',
                       absolute=True, ensemble='isometric',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_A(v, approach='monte carlo',
                       absolute=True, ensemble='isotensional',
                       num_processes=1, num_burns=8, num_samples=8)

    def test_interface_relative_nondimensional_gibbs_free_energy(self):
        """Function to test relative Gibbs free energy interface.

        """
        beta_G = random_crack_model().beta_G
        for p in 1e-1*np.random.rand(), 1e-1*np.random.rand(8):
            _ = beta_G(p)
            _ = beta_G(p, ensemble='isometric')
            _ = beta_G(p, ensemble='isotensional')
            _ = beta_G(p, approach='asymptotic')
            _ = beta_G(p, absolute=False)
            _ = beta_G(p, absolute=True)
            _ = beta_G(p, ensemble='isometric', approach='asymptotic')
            _ = beta_G(p, ensemble='isotensional', approach='asymptotic')
            _ = beta_G(p, ensemble='isometric', absolute=False)
            _ = beta_G(p, ensemble='isotensional', absolute=False)
            _ = beta_G(p, ensemble='isometric', absolute=True)
            _ = beta_G(p, ensemble='isotensional', absolute=True)
            _ = beta_G(p, absolute=False, approach='asymptotic')
            _ = beta_G(p, absolute=True, approach='asymptotic')
            _ = beta_G(p, ensemble='isometric',
                       approach='asymptotic', absolute=False)
            _ = beta_G(p, ensemble='isometric',
                       approach='asymptotic', absolute=False)
            _ = beta_G(p, ensemble='isotensional',
                       approach='asymptotic', absolute=True)
            _ = beta_G(p, ensemble='isotensional',
                       approach='asymptotic', absolute=True)
            _ = beta_G(p, approach='monte carlo',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo', ensemble='isometric',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo', ensemble='isotensional',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo',
                       absolute=False,
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo',
                       absolute=False, ensemble='isometric',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo',
                       absolute=False, ensemble='isotensional',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo',
                       absolute=True,
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo',
                       absolute=True, ensemble='isometric',
                       num_processes=1, num_burns=8, num_samples=8)
            _ = beta_G(p, approach='monte carlo',
                       absolute=True, ensemble='isotensional',
                       num_processes=1, num_burns=8, num_samples=8)

    def test_interface_relative_nondimensional_reaction_rate_coefficient(self):
        """Function to test relative nondimensional
        reaction rate coefficient interface.

        """
        k = random_crack_model().k
        for v in 1 + 1e-1*np.random.rand(), 1 + 1e-1*np.random.rand(8):
            _ = k(v)
            _ = k(v, ensemble='isometric')
            _ = k(v, approach='asymptotic')
            _ = k(v, ensemble='isometric', approach='asymptotic')
            _ = k(v, approach='monte carlo',
                  num_processes=1, num_burns=8, num_samples=8)
            _ = k(v, approach='monte carlo', ensemble='isometric',
                  num_processes=1, num_burns=8, num_samples=8)
        for p in 1e-1*np.random.rand(), 1e-1*np.random.rand(8):
            _ = k(p, ensemble='isotensional')
            _ = k(p, ensemble='isotensional', approach='asymptotic')
            _ = k(p, approach='monte carlo', ensemble='isotensional',
                  num_processes=1, num_burns=8, num_samples=8)


if __name__ == '__main__':
    unittest.main()
