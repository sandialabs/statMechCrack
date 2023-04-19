"""A module for the quasi-two-dimensional crack model
in the isotensional ensemble.

"""

import numpy as np
from numpy.linalg import det, inv

from .isometric import CrackQ2DIsometric


class CrackQ2DIsotensional(CrackQ2DIsometric):
    """The quasi-two-dimensional crack model class
    for the isotensional ensemble.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackQ2DIsotensional` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackQ2DIsometric` class instance.

        """
        CrackQ2DIsometric.__init__(self, **kwargs)

    def k_isotensional(self, p, transition_state):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end forces
        in the isotensional ensemble.

        Args:
            p (array_like): The nondimensional end forces.
            transition_state (int): The kth transition state.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        p_ref = np.zeros(self.W)
        beta_Pi, _, _, hess = self.minimize_beta_Pi(p)
        beta_Pi_ref, _, _, hess_ref = self.minimize_beta_Pi(p_ref)
        beta_Pi_TS, _, _, hess_TS = self.minimize_beta_Pi(
            p, transition_state=transition_state
        )
        beta_Pi_TS_ref, _, _, hess_TS_ref = self.minimize_beta_Pi(
            p_ref, transition_state=transition_state
        )
        scale = self.varepsilon*(self.W*self.L)**(1/3)
        return np.exp(
            beta_Pi - beta_Pi_ref - beta_Pi_TS + beta_Pi_TS_ref
        )*np.sqrt(
            det(hess/scale) /
            det(hess_ref/scale) *
            det(hess_TS_ref/scale) /
            det(hess_TS/scale)
        )
