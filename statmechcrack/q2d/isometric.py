"""A module for the quasi-two-dimensional crack model
in the isometric ensemble.

"""

import numpy as np
from numpy.linalg import det, inv

from .mechanical import CrackQ2DMechanical


class CrackQ2DIsometric(CrackQ2DMechanical):
    """The quasi-two-dimensional crack model class
    for the isometric ensemble.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackQ2DIsometric` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackQ2DMechanical` class instance.

        """
        CrackQ2DMechanical.__init__(self, **kwargs)

    def k_isometric(self, v, transition_state):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end separations
        in the isometric ensemble.

        Args:
            v (array_like): The nondimensional end separations.
            transition_state (int): The kth transition state.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        v_ref = np.ones(self.W)
        beta_U, _, _, hess = self.minimize_beta_U(v)
        beta_U_ref, _, _, hess_ref = self.minimize_beta_U(v_ref)
        beta_U_TS, _, _, hess_TS = self.minimize_beta_U(
            v, transition_state=transition_state
        )
        beta_U_TS_ref, _, _, hess_TS_ref = self.minimize_beta_U(
            v_ref, transition_state=transition_state
        )
        return np.exp(
            beta_U - beta_U_ref - beta_U_TS + beta_U_TS_ref
        )*np.sqrt(
            det(hess.dot(inv(hess_ref))) /
            det(hess_TS.dot(inv(hess_TS_ref)))
        )
