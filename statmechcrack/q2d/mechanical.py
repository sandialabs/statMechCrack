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
        self, L=16, N=8*np.ones(8, dtype=int), W=8,
        kappa=100, alpha=1, varepsilon=100,
        periodic_boundary_conditions=False
    ):
        """Initializes the :class:`CrackQ2DMechanical` class.

        Initialize and inherit all attributes and methods
        from a :class:`BasicUtility` class instance.

        """
        assert(len(N) == W)
        BasicUtility.__init__(self)
        self.N = N
        self.L = L
        self.M = L - N
        self.W = W
        self.kappa = kappa
        self.alpha = alpha
        self.varepsilon = varepsilon
        self.lambda_TS = 1 + np.log(2)/alpha
        self.periodic_boundary_conditions = periodic_boundary_conditions
        self.__p_mech_helper = (
            np.diag(6*np.ones(self.W)) +
            np.diag(-4*np.ones(self.W - 1), 1) +
            np.diag(-4*np.ones(self.W - 1), -1) +
            np.diag(np.ones(self.W - 2), 2) +
            np.diag(np.ones(self.W - 2), -2)
        )
        if self.periodic_boundary_conditions is True:
            self.__p_mech_helper[-1, 0] = -4
            self.__p_mech_helper[0, -1] = -4
            self.__p_mech_helper[-2, 0] = 1
            self.__p_mech_helper[0, -2] = 1
            self.__p_mech_helper[-1, 1] = 1
            self.__p_mech_helper[1, -1] = 1
        else:
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
        self.__H_U_0 = np.squeeze(
            beta_U_0.diff(s_vec, 2)
        ).astype(np.float64)
        p = symarray('p', self.W)
        beta_Pi_0 = self.beta_Pi_0(p, v, s)
        vs = np.concatenate(([v], s))
        vs_vec = Matrix(np.resize(vs, (self.L + 1)*self.W))
        self.__H_Pi_0 = np.squeeze(
            beta_Pi_0.diff(vs_vec, 2)
        ).astype(np.float64)

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
            \right)^2,

        where :math:`s_0^k\equiv v_k`.
        For periodic boundary conditions,

        .. math::
            \beta U_0(\mathbf{v}, \mathbf{s}) =
            \sum_{i=2}^{L} \sum_{k=1}^{W} \frac{\kappa}{2} \left(
                s_{i-2}^k - 2s_{i-1}^k + s_i^k
            \right)^2 +
            \sum_{i=0}^{L} \sum_{k=3}^{W} \frac{\kappa}{2} \left(
                s_i^{k-2} - 2s_i^{k-1} + s_i^k
            \right)^2 +
            \sum_{i=0}^{L} \frac{\kappa}{2} \left[
                \left(
                    s_i^{W-1} - 2s_i^W + s_i^1
                \right)^2 +
                \left(
                    s_i^{W} - 2s_i^1 + s_i^2
                \right)^2
            \right].

        Args:
            v (array_like): The nondimensional end separations.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        vs = np.concatenate(([v], s))
        beta_U_0 = self.kappa/2*(
            np.sum(np.diff(vs, 2, axis=0)**2) +
            np.sum(np.diff(vs, 2, axis=1)**2)
        )
        if self.periodic_boundary_conditions is True:
            beta_U_0 += self.kappa/2*(
                np.sum((vs[:, -2] - 2*vs[:, -1] + vs[:, 0])**2) +
                np.sum((vs[:, -1] - 2*vs[:, 0] + vs[:, 1])**2)
            )
        return beta_U_0

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

    def beta_U(self, v, s_vec, transition_state=None):
        r"""The nondimensional potential energy of the system,

        .. math::
            \beta U(\mathbf{v},\mathbf{s}) =
            \beta U_0(\mathbf{v},\mathbf{s}) + \beta U_1(\boldsymbol{\lambda}),

        where :math:`\lambda_j^k\equiv s_{N_j+j}^k` and :math:`j=1,\ldots,M`.

        Args:
            v (array_like): The nondimensional end separations.
            s_vec (array_like): The nondimensional configuration.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        if transition_state is not None:
            s_vec = np.insert(
                s_vec,
                np.ravel_multi_index(
                    (self.N[transition_state], transition_state),
                    (self.L, self.W)
                ),
                self.lambda_TS
            )
        s = np.reshape(s_vec, (self.L, self.W))
        beta_U = self.beta_U_0(v, s)
        for k in range(self.W):
            beta_U += self.beta_U_1(s[-self.M[k]:, k])
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
        for k in range(self.W):
            lambda_[:self.N[k], k] = 1
        lambda_vec = np.reshape(lambda_, self.L*self.W)
        return self.beta_u_p(lambda_vec)

    def j_U(self, v, s_vec, transition_state=None):
        r"""The nondimensional Jacobian
        of the potential energy of the system.

        Args:
            v (array_like): The nondimensional end separations.
            s_vec (array_like): The nondimensional configuration.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        if transition_state is not None:
            s_vec = np.insert(
                s_vec,
                np.ravel_multi_index(
                    (self.N[transition_state], transition_state),
                    (self.L, self.W)
                ),
                self.lambda_TS
            )
        j_U = self.j_U_0(v, s_vec) + self.j_U_1(s_vec)
        if transition_state is not None:
            j_U = np.delete(
                j_U,
                np.ravel_multi_index(
                    (self.N[transition_state], transition_state),
                    (self.L, self.W)
                )
            )
        return j_U

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
        beta_u_pp = self.beta_u_pp(lambda_)
        for k in range(self.W):
            beta_u_pp[:self.N[k], k] = 0
        return np.diag(np.reshape(beta_u_pp, self.L*self.W))

    def H_U(self, s_vec, transition_state=None):
        r"""The nondimensional Hessian
        of the potential energy of the system.

        Args:
            s_vec (array_like): The nondimensional configuration.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        if transition_state is not None:
            s_vec = np.insert(
                s_vec,
                np.ravel_multi_index(
                    (self.N[transition_state], transition_state),
                    (self.L, self.W)
                ),
                self.lambda_TS
            )
        H_U = self.H_U_0() + self.H_U_1(s_vec)
        if transition_state is not None:
            index = np.ravel_multi_index(
                (self.N[transition_state], transition_state),
                (self.L, self.W)
            )
            H_U = np.delete(np.delete(H_U, index, axis=0), index, axis=1)
        return H_U

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

    def beta_Pi(self, p, vs_vec, transition_state=None):
        r"""The nondimensional total potential energy of the system,

        .. math::
            \beta \Pi(\mathbf{p},\mathbf{v},\mathbf{s}) =
            \beta U(\mathbf{v},\mathbf{s}) - \mathbf{p}\cdot\mathbf{v}.

        Args:
            p (array_like): The nondimensional end forces.
            vs_vec (array_like): The nondimensional configuration.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            numpy.ndarray: The nondimensional total potential energy.

        """
        if transition_state is not None:
            vs_vec = np.insert(
                vs_vec,
                np.ravel_multi_index(
                    (self.N[transition_state] + 1, transition_state),
                    (self.L + 1, self.W)
                ),
                self.lambda_TS
            )
        vs = np.reshape(vs_vec, (self.L + 1, self.W))
        v = vs[0, :]
        s = vs[1:, :]
        beta_U = self.beta_U_0(v, s)
        for k in range(self.W):
            beta_U += self.beta_U_1(s[-self.M[k]:, k])
        return beta_U - p.dot(v)

    def j_Pi_0(self, p, vs_vec):
        r"""The nondimensional Jacobian
        of the total potential energy of the system
        for the reference system.

        Args:
            p (array_like): The nondimensional end forces.
            vs_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        return self.H_Pi_0().dot(vs_vec) - np.reshape(
            np.concatenate(
                ([p], np.zeros((self.L, self.W)))
            ), (self.L + 1)*self.W
        )

    def j_Pi_1(self, vs_vec):
        r"""The nondimensional Jacobian
        of the total potential energy of the system
        due to stretching intact bonds.

        Args:
            vs_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        lambda_ = np.reshape(vs_vec, (self.L + 1, self.W))
        for k in range(self.W):
            lambda_[:self.N[k] + 1, k] = 1
        lambda_vec = np.reshape(lambda_, (self.L + 1)*self.W)
        return self.beta_u_p(lambda_vec)

    def j_Pi(self, p, vs_vec, transition_state=None):
        r"""The nondimensional Jacobian
        of the total potential energy of the system.

        Args:
            p (array_like): The nondimensional end forces.
            vs_vec (array_like): The nondimensional configuration.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        if transition_state is not None:
            vs_vec = np.insert(
                vs_vec,
                np.ravel_multi_index(
                    (self.N[transition_state] + 1, transition_state),
                    (self.L + 1, self.W)
                ),
                self.lambda_TS
            )
        j_Pi = self.j_Pi_0(p, vs_vec) + self.j_Pi_1(vs_vec)
        if transition_state is not None:
            j_Pi = np.delete(
                j_Pi,
                np.ravel_multi_index(
                    (self.N[transition_state] + 1, transition_state),
                    (self.L + 1, self.W)
                )
            )
        return j_Pi

    def H_Pi_0(self):
        r"""The nondimensional Hessian
        of the total potential energy of the system
        due to bending.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return self.__H_Pi_0

    def H_Pi_1(self, vs_vec):
        r"""The nondimensional Hessian
        of the total potential energy of the system
        due to stretching intact bonds.

        Args:
            vs_vec (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        lambda_ = np.reshape(vs_vec, (self.L + 1, self.W))
        beta_u_pp = self.beta_u_pp(lambda_)
        for k in range(self.W):
            beta_u_pp[:self.N[k] + 1, k] = 0
        return np.diag(np.reshape(beta_u_pp, (self.L + 1)*self.W))

    def H_Pi(self, vs_vec, transition_state=None):
        r"""The nondimensional Hessian
        of the total potential energy of the system.

        Args:
            p (array_like): The nondimensional end forces.
            vs_vec (array_like): The nondimensional configuration.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        if transition_state is not None:
            vs_vec = np.insert(
                vs_vec,
                np.ravel_multi_index(
                    (self.N[transition_state] + 1, transition_state),
                    (self.L + 1, self.W)
                ),
                self.lambda_TS
            )
        H_Pi = self.H_Pi_0() + self.H_Pi_1(vs_vec)
        if transition_state is not None:
            index = np.ravel_multi_index(
                (self.N[transition_state] + 1, transition_state),
                (self.L + 1, self.W)
            )
            H_Pi = np.delete(np.delete(H_Pi, index, axis=0), index, axis=1)
        return H_Pi

    def minimize_beta_U(self, v, transition_state=None):
        r"""Function to minimize the potential energy of the system.

        Args:
            v (array_like): The nondimensional end separations.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional forces.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional configuration.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional Hessian.

        """
        s_vec_guess = np.ones(
            self.L*self.W - (transition_state is not None)
        )
        res = minimize(
            lambda s_vec: self.beta_U(
                v, s_vec, transition_state=transition_state
            ),
            s_vec_guess,
            method='Newton-CG',
            jac=lambda s_vec: self.j_U(
                v, s_vec, transition_state=transition_state
            ),
            hess=lambda s_vec: self.H_U(
                s_vec, transition_state=transition_state
            )
        )
        s_vec = res.x
        if transition_state is not None:
            s_vec = np.insert(
                s_vec,
                np.ravel_multi_index(
                    (self.N[transition_state], transition_state),
                    (self.L, self.W)
                ),
                self.lambda_TS
            )
        s = np.resize(s_vec, (self.L, self.W))
        p = self.kappa*(v - 2*s[0, :] + s[1, :] + self.__p_mech_helper.dot(v))
        beta_U = res.fun
        H_U = self.H_U(res.x, transition_state=transition_state)
        return beta_U, p, s, H_U

    def minimize_beta_Pi(self, p, transition_state=None):
        r"""Function to minimize the total potential energy of the system.

        Args:
            p (array_like): The nondimensional end forces.
            transition_state (int, optional, default=None):
                Whether or not the system is in the kth transition state.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional separations.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional configuration.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional Hessian.

        """
        vs_vec_guess = np.ones(
            (self.L + 1)*self.W - (transition_state is not None)
        )
        res = minimize(
            lambda vs_vec: self.beta_Pi(
                p, vs_vec, transition_state=transition_state
            ),
            vs_vec_guess,
            method='Newton-CG',
            jac=lambda vs_vec: self.j_Pi(
                p, vs_vec, transition_state=transition_state
            ),
            hess=lambda vs_vec: self.H_Pi(
                vs_vec, transition_state=transition_state
            )
        )
        vs_vec = res.x
        if transition_state is not None:
            vs_vec = np.insert(
                vs_vec,
                np.ravel_multi_index(
                    (self.N[transition_state] + 1, transition_state),
                    (self.L + 1, self.W)
                ),
                self.lambda_TS
            )
        vs = np.resize(vs_vec, (self.L + 1, self.W))
        v = vs[0, :]
        s = vs[1:, :]
        beta_U = res.fun
        H_Pi = self.H_Pi(res.x, transition_state=transition_state)
        return beta_U, v, s, H_Pi

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

    def v_mechanical(self, p):
        r"""The nondimensional end separations
        as a function of the nondimensional end forces
        for the mechnically-treated system,
        calculated by minimizing the potential energy.

        Args:
            p (array_like): The nondimensional end forces.

        Returns:
            numpy.ndarray: The nondimensional end separations.

        """
        return self.minimize_beta_Pi(p)[1]
