"""A module for the quasi-two-dimensional crack model treated mechanically.

"""

import numpy as np
from scipy.optimize import minimize
from sympy import Matrix, symarray

from ..utility import BasicUtility


class CrackQ2DMechanical(BasicUtility):
    """The quasi-two-dimensional crack model class treated mechanically.

    """
    def __init__(
        self, L=16, N=8*np.ones(9), W=9, kappa=100, alpha=1, varepsilon=100
    ):
        """Initializes the :class:`CrackQ2DMechanical` class.

        Initialize and inherit all attributes and methods
        from a :class:`BasicUtility` class instance.

        """

# N and M need to be W-length vectors

# assert N/M consistent with W here, other assertions

# shouldnt you force W to be odd? like maybe k=0 at center and +/- 1 either side? and then W on either side of the zero?

        BasicUtility.__init__(self)
        assert(len(N) == W)
        self.N = N
        self.L = L
        self.M = L - N
        self.W = W
        self.kappa = kappa
        self.alpha = alpha
        self.varepsilon = varepsilon
        self.lambda_TS = 1 + np.log(2)/alpha
        self.__p_mech_helper = (
            np.diag(6*np.ones(self.W)) +
            np.diag(-4*np.ones(self.W - 1), 1) +
            np.diag(-4*np.ones(self.W - 1), -1) +
            np.diag(np.ones(self.W - 2), 2) +
            np.diag(np.ones(self.W - 2), -2)
        )
        self.__p_mech_helper[0, 0] = 1
        self.__p_mech_helper[-1, -1] = 1
        self.__p_mech_helper[1, 1] = 5
        self.__p_mech_helper[-2, -2] = 5
        self.__p_mech_helper[0, 1] = -2
        self.__p_mech_helper[1, 0] = -2
        self.__p_mech_helper[-1, -2] = -2
        self.__p_mech_helper[-2, -1] = -2
        v = symarray('v', self.W)
        s = symarray('s', (self.L, self.W))
        s_vec = Matrix(np.resize(s, self.L*self.W))
        beta_U_0 = self.beta_U_0(v, s)
        self.__H_U_0 = np.squeeze(beta_U_0.diff(s_vec, 2)).astype(np.float64)

    def beta_U_0(self, v, s):
        r"""The nondimensional potential energy of the system
        due to bending, i.e. of the reference system,

        .. math::
            \beta U_0(\mathbf{v}, \mathbf{s}) =
            \sum_{i=2}^{L} \sum_{k=1}^{W} \frac{\kappa}{2} \left(
                s_{i-2}^k - 2s_{i-1}^k + s_i^k
            \right)^2 +
            \sum_{i=0}^{L} \sum_{k=3}^{W} \frac{\kappa}{2} \left(
                s_i^{k-2} - 2s_i^{k-1} + s_i^k
            \right)^2

        where :math:`s_0^k\equiv v_k`.

        Args:
            v (array_like): The nondimensional end separations.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        vs = np.concatenate(([v], s))
        return self.kappa/2*(
            np.sum(np.diff(vs, 2, axis=0)**2) +
            np.sum(np.diff(vs, 2, axis=1)**2)
        )

    def beta_u(self, lambda_):
        r"""The nondimensional potential energy of a single bond
        as a function of the bond stretch, given by the
        Morse potential :footcite:`morse1929diatomic`,

        .. math::
            \beta u(\lambda) =
            \varepsilon \left[
                1 - e^{-\alpha(\lambda - 1)}
            \right]^2.

        Args:
            lambda_ (array_like): The bond stretch.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        return self.varepsilon*(
            1 - np.exp(-self.alpha*(lambda_ - 1))
        )**2 + 1e88*(lambda_ > self.lambda_TS)

    def beta_u_p(self, lambda_):
        r"""The first derivative of the potential energy of a single bond
        as a function of the bond stretch,

        .. math::
            \beta u'(\lambda) =
            2\alpha\varepsilon e^{-\alpha(\lambda - 1)} \left[
                1 - e^{-\alpha(\lambda - 1)}
            \right]^2.

        Args:
            lambda_ (array_like): The bond stretch.

        Returns:
            numpy.ndarray:
                The first derivative of the nondimensional potential energy.

        """
        return 2*self.alpha*self.varepsilon * \
            np.exp(-self.alpha*(lambda_ - 1)) * \
            (1 - np.exp(-self.alpha*(lambda_ - 1)))

    def beta_u_pp(self, lambda_):
        r"""The second derivative of the potential energy of a single bond
        as a function of the bond stretch,

        .. math::
            \beta u''(\lambda) =
            2\alpha^2\varepsilon e^{-\alpha(\lambda - 1)} \left[
                2e^{-\alpha(\lambda - 1)} - 1
            \right]^2.

        Args:
            lambda_ (array_like): The bond stretch.

        Returns:
            numpy.ndarray:
                The second derivative of the nondimensional potential energy.

        """
        return 2*self.alpha**2*self.varepsilon * \
            np.exp(-self.alpha*(lambda_ - 1)) * \
            (2*np.exp(-self.alpha*(lambda_ - 1)) - 1)

    def beta_U_1(self, lambda_):
        r"""The nondimensional potential energy of the system
        due to stretching intact bonds.

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        return np.sum(self.beta_u(lambda_))

    def beta_U(self, v, s):
        r"""The nondimensional potential energy of the system,

        .. math::
            \beta U(\mathbf{v},\mathbf{s}) =
            \beta U_0(\mathbf{v},\mathbf{s}) + \beta U_1(\boldsymbol{\lambda}),

        where :math:`\lambda_j^k\equiv s_{N_j+j}^k` and :math:`j=1,\ldots,M`.

        Args:
            v (array_like): The nondimensional end separations.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        beta_U = self.beta_U_0(v, s)
        for j in range(self.W):
            beta_U += self.beta_U_1(s[-self.M[j]:, j])
        return beta_U

    def j_U_0(self, v, s_vec):
        r"""The nondimensional Jacobian
        of the potential energy of the system
        for the reference system.

        Args:
            v (array_like): The nondimensional end separations.
            s_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        j_U_0 = np.zeros((self.L, self.W))
        j_U_0[0, :] += -2*self.kappa*v
        j_U_0[1, :] += self.kappa*v
        j_U_0.resize(self.L*self.W)
        j_U_0 += self.H_U_0().dot(s_vec)
        return j_U_0

    def j_U_1(self, s_vec):
        r"""The nondimensional Jacobian
        of the potential energy of the system
        due to stretching intact bonds.

        Args:
            s_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        lambda_ = np.reshape(s_vec, (self.L, self.W))
        for j in range(self.W):
            lambda_[:-self.M[j], j] = 1
        lambda_vec = np.reshape(lambda_, self.L*self.W)
        return self.beta_u_p(lambda_vec)

    def j_U(self, v, s_vec):
        r"""The nondimensional Jacobian
        of the potential energy of the system.

        Args:
            v (array_like): The nondimensional end separations.
            s_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        return self.j_U_0(v, s_vec) + self.j_U_1(s_vec)

    def H_U_0(self):
        r"""The nondimensional Hessian
        of the potential energy of the system
        due to bending.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return self.__H_U_0

    def H_U_1(self, s_vec):
        r"""The nondimensional Hessian
        of the potential energy of the system
        due to stretching intact bonds.

        Args:
            s_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        lambda_ = np.reshape(s_vec, (self.L, self.W))
        H_U_1 = self.beta_u_pp(lambda_)
        for j in range(self.W):
            H_U_1[:-self.M[j], j] = 0
        H_U_1 = np.reshape(H_U_1, self.L*self.W)
        return H_U_1

    def H_U(self, s_vec):
        r"""The nondimensional Hessian
        of the potential energy of the system.

        Args:
            s_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return self.H_U_0() + self.H_U_1(s_vec)

    def beta_Pi_0(self, p, v, s):
        r"""The nondimensional total potential energy
        of the reference system,

        .. math::
            \beta \Pi_{0}(\mathbf{p},\mathbf{v},\mathbf{s}) =
            \beta U_{0}(\mathbf{v},\mathbf{s}) - \mathbf{p}\cdot\mathbf{v}.

        Args:
            p (array_like): The nondimensional end forces.
            v (array_like): The nondimensional end separations.
            s (array_like): The nondimensional configuration.

        """
        return self.beta_U_0(v, s) - p.dot(v)

    def beta_Pi(self, p, v, s):
        r"""The nondimensional total potential energy of the system,

        .. math::
            \beta \Pi(\mathbf{p},\mathbf{v},\mathbf{s}) =
            \beta U(\mathbf{v},\mathbf{s}) - \mathbf{p}\cdot\mathbf{v}.

        Args:
            p (array_like): The nondimensional end forces.
            v (array_like): The nondimensional end separations.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional total potential energy.

        """
        return self.beta_U(v, s) - p.dot(v)

    def minimize_beta_U(self, v):
        r"""Function to minimize the potential energy of the system.

        Args:
            v (array_like): The nondimensional end separations.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional forces.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional configuration.

        """
        s_guess = np.resize(np.ones((self.L, self.W)), self.L*self.W)
        res = minimize(
            lambda s: self.beta_U(v, np.resize(s, (self.L, self.W))),
            s_guess,
            method='Newton-CG',
            jac=lambda s: self.j_U(v, s),
            hess=self.H_U
        )
        s = np.resize(res.x, (self.L, self.W))
        p = self.kappa*(v - 2*s[0, :] + s[1, :] + self.__p_mech_helper.dot(v))
        beta_U = res.fun
        return beta_U, p, s

    def p_mechanical(self, v):
        r"""The nondimensional end forces
        as a function of the nondimensional end separations
        for the mechnically-treated system,
        calculated by minimizing the potential energy.

        Args:
            v (array_like): The nondimensional end separations.

        Returns:
            numpy.ndarray: The nondimensional end forces.

        """
        return self.minimize_beta_U(v)[1]
