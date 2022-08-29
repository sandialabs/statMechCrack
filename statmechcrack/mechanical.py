"""A module for the crack model treated mechanically.

This module consist of the class :class:`CrackMechanical` which
contains methods for computing quantities when treating
the crack model mechanically rather than statistical mechanically.

"""

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize

from .utility import BasicUtility


class CrackMechanical(BasicUtility):
    """The crack model class for the isometric ensemble.

    Attributes:
        N (int): The number of broken bonds.
        M (int): The number of intact bonds.
        L (int): The total number of bonds.
        kappa (float): The nondimensional bending stiffness.
        alpha (float): The nondimensional Morse parameter.
        varepsilon (float): The nondimensional bond energy.

    """
    def __init__(self, N=8, M=8, kappa=100, alpha=1, varepsilon=100):
        """Initializes the :class:`CrackMechanical` class.

        Initialize and inherit all attributes and methods
        from a :class:`BasicUtility` class instance.

        """
        BasicUtility.__init__(self)
        self.N = N
        self.M = M
        self.L = N + M
        self.kappa = kappa
        self.alpha = alpha
        self.varepsilon = varepsilon
        self.lambda_TS = 1 + np.log(2)/alpha
        self.v_TS = minimize(
            lambda v: (
                self.beta_u(
                    self.minimize_beta_U(v)[2][-self.M]
                )
                - self.beta_u(self.lambda_TS)
            )**2,
            self.lambda_TS
        ).x[0]
        self.p_TS = self.p_mechanical(self.v_TS)[0]

    def beta_U_00(self, v, s):
        r"""The nondimensional potential energy
        of the isolated bending system,

        .. math::
            \beta U_{00}(v, \mathbf{s}) =
            \sum_{i=1}^{N+1} \frac{\kappa}{2} \left(
                s_{i-1} - 2s_i + s_{i+1}
            \right)^2,

        where :math:`s_0\equiv v`, and
        :math:`s_{N+1}\equiv\lambda_1`, and
        :math:`s_{N+2}\equiv\lambda_2`.

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        vs_b = np.append([v], s[:self.N + 1])
        return self.kappa/2*np.sum(np.diff(vs_b, 2)**2)

    def beta_U_01(self, lambda_):
        r"""The nondimensional potential energy
        due to bending within the intact region,

        .. math::
            \beta U_{01}(\boldsymbol{\lambda})
            &=
            \beta U_0(v, \mathbf{s}) - \beta U_{00}(v, \mathbf{s})
            \\ &=
            \sum_{i=N+2}^{L-1} \frac{\kappa}{2} \left(
                s_{i-1} - 2s_i + s_{i+1}
            \right)^2
            \\ &=
            \sum_{j=2}^{M-1} \frac{\kappa}{2} \left(
                \lambda_{j-1} - 2\lambda_j + \lambda_{j+1}
            \right)^2,

        where :math:`s_0\equiv v`, and
        :math:`s_{N+1}\equiv\lambda_1`, and
        :math:`s_{N+2}\equiv\lambda_2`.

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        return self.kappa/2*np.sum(np.diff(self.np_array(lambda_), 2)**2)

    def beta_U_0(self, v, s):
        r"""The nondimensional potential energy of the system
        due to bending, i.e. of the reference system,

        .. math::
            \beta U_0(v, \mathbf{s}) =
            \sum_{i=1}^{L-1} \frac{\kappa}{2} \left(
                s_{i-1} - 2s_i + s_{i+1}
            \right)^2,

        where :math:`s_0\equiv v`.

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        vs = np.append([v], s)
        return self.kappa/2*np.sum(np.diff(vs, 2)**2)

    def beta_u(self, lambda_):
        r"""The nondimensional potential energy of a single bond
        as a function of the bond stretch, given by the
        Morse potential :cite:`morse1929diatomic`,

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
        due to stretching intact bonds,

        .. math::
            \beta U_1(\boldsymbol{\lambda}) =
            \sum_{j=1}^M \varepsilon\beta u(\lambda_j).

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        return np.sum(self.beta_u(self.np_array(lambda_)))

    def beta_U(self, v, s):
        r"""The nondimensional potential energy of the system,

        .. math::
            \beta U(v,\mathbf{s}) =
            \beta U_0(v,\mathbf{s}) + \beta U_1(\boldsymbol{\lambda}),

        where :math:`\lambda_j\equiv s_{N+j}` and :math:`j=1,\ldots,M`.

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional potential energy.

        """
        lambda_ = s[-self.M:]
        return self.beta_U_0(v, s) + self.beta_U_1(lambda_)

    def j_U_00(self, v, s):
        r"""The nondimensional Jacobian
        of the potential energy of the system
        for the isolated bending system,

        .. math::
            \mathbf{j}^U_00(v,\mathbf{s}) =
            \mathbf{H}^U_00 \cdot \mathbf{s} - \kappa\left(
                2v, -v, 0, \ldots, 0, -\lambda_1, 4\lambda_1-\lambda_2
            \right)^T.

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        lambda_ = s[-self.M:]
        return self.H_U_00().dot(s[:self.N]) + \
            - self.kappa*np.concatenate((
                [2*v], [-v], np.zeros(self.N - 4),
                [-lambda_[0], 4*lambda_[0] - lambda_[1]]
            ))

    def j_U_0(self, v, s):
        r"""The nondimensional Jacobian
        of the potential energy of the system
        for the reference system,

        .. math::
            \mathbf{j}^U_0(v,\mathbf{s}) =
            \mathbf{H}^U_0 \cdot \mathbf{s} - \kappa\left(
                2v, -v, 0, \ldots, 0
            \right)^T

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        return self.H_U_0().dot(s) + \
            - self.kappa*np.concatenate(([2*v], [-v], np.zeros(self.L - 2)))

    def j_U_1(self, lambda_):
        r"""The nondimensional Jacobian
        of the potential energy of the system
        due to stretching intact bonds,

        .. math::
            \left[\mathbf{j}^U_1(\boldsymbol{\lambda})\right]_j =
            2\alpha\varepsilon e^{-\alpha(\lambda_j-1)}\left(
                1 - e^{-\alpha(\lambda_j-1)}
            \right).

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        return np.concatenate((
            np.zeros(self.N),
            self.beta_u_p(lambda_)
        ))

    def j_U(self, v, s):
        r"""The nondimensional Jacobian
        of the potential energy of the system,

        .. math::
            \mathbf{j}^U(v,\mathbf{s}) =
            \mathbf{j}^U_0(v,\mathbf{s})
            +\mathbf{j}^U_1(\boldsymbol{\lambda}).

        where :math:`\lambda_j\equiv s_{N+j}` and :math:`j=1,\ldots,M`.

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        lambda_ = s[-self.M:]
        return self.j_U_0(v, s) + self.j_U_1(lambda_)

    def j_U_TS(self, v, s):
        r"""The nondimensional Jacobian
        of the potential energy of the system
        in its transition state.

        Args:
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        rows_cols = np.concatenate((
            np.arange(self.N), np.arange(self.N + 1, self.L)
        ))
        return self.j_U(v, np.concatenate((
            s[:self.N], [self.lambda_TS], s[-(self.M - 1):]
        )))[rows_cols]

    def H_U_00(self):
        r"""The nondimensional Hessian
        of the potential energy
        for the isolated bending system,

        .. math::
            \left[\mathbf{H}^U_00\right]_{mn} =
            \kappa\left(6\delta^m_n - \delta^m_1\delta^n_1
            - 4\delta^{|m-n|}_1 + \delta^{|m-n|}_2\right),
            \quad m,n=1,\ldots,N.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return self.H_Pi_00()[1:, 1:]

    def H_U_0(self):
        r"""The nondimensional Hessian
        of the potential energy of the system
        due to bending,

        .. math::
            \left[\mathbf{H}^U_0\right]_{mn} =
            \left[\mathbf{H}^\pi_0\right]_{m+1,n+1},
            \quad m,n=1,\ldots,L.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return self.H_Pi_0()[1:, 1:]

    def H_U_1(self, lambda_):
        r"""The nondimensional Hessian
        of the potential energy of the system
        due to stretching intact bonds,

        .. math::
            \left[\mathbf{H}^U_1(\boldsymbol{\lambda})\right]_{mn} =
            2\alpha^2\varepsilon e^{-\alpha(\lambda_m-1)}\left(
                1 - 2e^{-\alpha(\lambda_m-1)}
            \right)\delta^m_n.

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return np.diag(
            np.concatenate((
                np.zeros(self.N),
                self.beta_u_pp(lambda_)
            ))
        )

    def H_U(self, s):
        r"""The nondimensional Hessian
        of the potential energy of the system,

        .. math::
            \mathbf{H}^U(\mathbf{s}) =
            \mathbf{H}^U_0
            +\mathbf{H}^U_1(\boldsymbol{\lambda}).

        where :math:`\lambda_j\equiv s_{N+j}` and :math:`j=1,\ldots,M`.

        Args:
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        lambda_ = s[-self.M:]
        return self.H_U_0() + self.H_U_1(lambda_)

    def H_U_TS(self, s):
        r"""The nondimensional Hessian
        of the potential energy of the system
        in its transition state.

        Args:
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        rows_cols = np.concatenate((
            np.arange(self.N), np.arange(self.N + 1, self.L)
        ))
        return self.H_U(np.concatenate((
                s[:self.N], [self.lambda_TS], s[-(self.M - 1):]
            )))[rows_cols, :][:, rows_cols]

    def beta_Pi_00(self, p, v, s):
        r"""The nondimensional total potential energy
        for the isolated bending system,

        .. math::
            \beta \Pi_b(p,v,\mathbf{s}) =
            \beta U_{00}(v,\mathbf{s}) - pv.

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        """
        return self.beta_U_00(v, s) - p*v

    def beta_Pi_0(self, p, v, s):
        r"""The nondimensional total potential energy
        of the reference system,

        .. math::
            \beta \Pi_{0}(p,v,\mathbf{s}) =
            \beta U_{0}(v,\mathbf{s}) - pv.

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        """
        return self.beta_U_0(v, s) - p*v

    def beta_Pi(self, p, v, s):
        r"""The nondimensional total potential energy of the system,

        .. math::
            \beta \Pi(p,v,\mathbf{s}) =
            \beta U(v,\mathbf{s}) - pv.

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional total potential energy.

        """
        return self.beta_U(v, s) - p*v

    def j_Pi_00(self, p, v, s):
        r"""The nondimensional Jacobian
        of the total potential energy of the system
        for the isolated bending system,

        .. math::
            \mathbf{j}^\Pi_b(p,v,\mathbf{s}) =
            \mathbf{H}^\Pi_b \cdot \left(
                v, s_1, \cdots, s_N
            \right)^T - \kappa\left(
                p/\kappa, 0, \ldots, 0, -\lambda_1, 4\lambda_1-\lambda_2
            \right)^T

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        lambda_ = s[-self.M:]
        return self.H_Pi_00().dot(np.concatenate(([v], s[:self.N]))) + \
            - self.kappa*np.concatenate((
                [p/self.kappa], np.zeros(self.N - 2),
                [-lambda_[0], 4*lambda_[0] - lambda_[1]]
            ))

    def j_Pi_0(self, p, v, s):
        r"""The nondimensional Jacobian
        of the total potential energy of the system
        due to bending,

        .. math::
            \mathbf{j}^\Pi_0(p,v,\mathbf{s}) =
            \mathbf{H}^\Pi_0 \cdot \left(
                v, s_1, \cdots, s_L
            \right)^T - \left(
                p, 0, \ldots, 0
            \right)^T

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        return self.H_Pi_0().dot(np.concatenate(([v], s))) \
            - np.concatenate(([p], np.zeros(self.L)))

    def j_Pi_1(self, lambda_):
        r"""The nondimensional Jacobian
        of the total potential energy of the system
        due to stretching intact bonds,

        .. math::
            \mathbf{j}^\Pi_1(\boldsymbol{\lambda}) =
            \mathbf{j}^U_1(\boldsymbol{\lambda}).

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        return np.concatenate((
            np.zeros(self.N + 1),
            self.beta_u_p(lambda_)
        ))

    def j_Pi(self, p, v, s):
        r"""The nondimensional Jacobian
        of the total potential energy of the system,

        .. math::
            \mathbf{j}^\Pi(p,v,\mathbf{s}) =
            \mathbf{j}^\Pi_0(p,v,\mathbf{s})
            +\mathbf{j}^\Pi_1(\boldsymbol{\lambda}).

        where :math:`\lambda_j\equiv s_{N+j}` and :math:`j=1,\ldots,M`.

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        lambda_ = s[-self.M:]
        return self.j_Pi_0(p, v, s) + self.j_Pi_1(lambda_)

    def j_Pi_TS(self, p, v, s):
        r"""The nondimensional Jacobian
        of the total potential energy of the system
        in its transition state.

        Args:
            p (array_like): The nondimensional end force.
            v (array_like): The nondimensional end separation.
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Jacobian.

        """
        rows_cols = np.concatenate((
            np.arange(self.N + 1), np.arange(self.N + 2, self.L + 1)
        ))
        return self.j_Pi(p, v, np.concatenate((
            s[:self.N], [self.lambda_TS], s[-(self.M - 1):]
        )))[rows_cols]

    def H_Pi_00(self):
        r"""The nondimensional Hessian
        of the total potential energy
        for the isolated bending system,

        .. math::
            \left[\mathbf{H}^\Pi_b\right]_{mn} =
            \kappa\left(\phantom{\delta^{|m-n|}_1}\right. &
            6\delta^m_n - 5\delta^m_1\delta^n_1
            - \delta^m_2\delta^n_2 - 4\delta^{|m-n|}_1
            \\ &
            + 2\delta^m_1\delta^n_2 + 2\delta^m_2\delta^n_1
            + \delta^{|m-n|}_2
            \left.\phantom{\delta^{|m-n|}_1}\right),
            \quad m,n=1,\ldots,N+1.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        diag_0 = np.concatenate(([1, 5], 6*np.ones(self.N - 1)))
        diag_1 = np.concatenate(([-2], -4*np.ones(self.N - 1)))
        diag_2 = np.ones(self.N - 1)
        return self.kappa*(np.diag(diag_0)
                           + np.diag(diag_1, 1) + np.diag(diag_1, -1)
                           + np.diag(diag_2, 2) + np.diag(diag_2, -2))

    def H_Pi_0(self):
        r"""The nondimensional Hessian
        of the total potential energy of the system
        due to bending,

        .. math::
            \left[\mathbf{H}^\Pi_0\right]_{mn} =
            \kappa\left(\phantom{\delta^{|m-n|}_1}\right. &
            6\delta^m_n - 5\delta^m_1\delta^n_1
            - \delta^m_2\delta^n_2 - 5\delta^m_{L+1}\delta^n_{L+1}
            - \delta^m_L\delta^n_L - 4\delta^{|m-n|}_1
            \\ &
            + 2\delta^m_1\delta^n_2 + 2\delta^m_2\delta^n_1
            + 2\delta^m_{L+1}\delta^n_L + 2\delta^m_L\delta^n_{L+1}
            + \delta^{|m-n|}_2
            \left.\phantom{\delta^{|m-n|}_1}\right),
            \\ &
            \quad m,n=1,\ldots,L+1.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        diag_0 = np.concatenate(([1, 5], 6*np.ones(self.L - 3), [5, 1]))
        diag_1 = np.concatenate(([-2], -4*np.ones(self.L - 2), [-2]))
        diag_2 = np.ones(self.L - 1)
        return self.kappa*(np.diag(diag_0)
                           + np.diag(diag_1, 1) + np.diag(diag_1, -1)
                           + np.diag(diag_2, 2) + np.diag(diag_2, -2))

    def H_Pi_1(self, lambda_):
        r"""The nondimensional Hessian
        of the total potential energy of the system
        due to stretching intact bonds,

        .. math::
            \left[\mathbf{H}^\Pi_1(\boldsymbol{\lambda})\right]_{mn} =
            \left[\mathbf{H}^U_1(\boldsymbol{\lambda})\right]_{mn}.

        Args:
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        return np.diag(
            np.concatenate((
                np.zeros(self.N + 1),
                self.beta_u_pp(lambda_)
            ))
        )

    def H_Pi(self, s):
        r"""The nondimensional Hessian
        of the total potential energy of the system,

        .. math::
            \mathbf{H}^\Pi(\mathbf{s}) =
            \mathbf{H}^\Pi_0
            +\mathbf{H}^\Pi_1(\boldsymbol{\lambda}).

        where :math:`\lambda_j\equiv s_{N+j}` and :math:`j=1,\ldots,M`.

        Args:
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        lambda_ = s[-self.M:]
        return self.H_Pi_0() + self.H_Pi_1(lambda_)

    def H_Pi_TS(self, s):
        r"""The nondimensional Hessian
        of the total potential energy of the system
        in its transition state.

        Args:
            s (array_like): The nondimensional configuration.

        Returns:
            numpy.ndarray: The nondimensional Hessian.

        """
        rows_cols = np.concatenate((
            np.arange(self.N + 1), np.arange(self.N + 2, self.L + 1)
        ))
        return self.H_Pi(np.concatenate((
            s[:self.N], [self.lambda_TS], s[-(self.M - 1):]
        )))[rows_cols, :][:, rows_cols]

    def minimize_beta_U_00(self, v, lambda_):
        r"""Function to minimize the potential energy
        of the isolated bending system.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional end separation.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional positions.

        Example:
            Plot the rescaled minimized nondimensional potential energy
            as a function of the nondimensional end separation
            for the mechnically-treated isolated bending system
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackMechanical
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25]:
                ...     model = CrackMechanical(N=N)
                ...     beta_U = model.minimize_beta_U_00(v, [1, 1])[0]
                ...     _ = plt.plot(v - 1, model.N**3/3/model.kappa*beta_U,
                ...                  label='$N=$'+str(N))
                ...     p_m = model.p_0_mechanical(v, [1, 1])
                ...     beta_Pi = model.minimize_beta_Pi_00(p_m, [1, 1])[0]
                ...     _ = plt.plot(v - 1,
                ...                  model.N**3/3/model.kappa*(beta_Pi+p_m*v)
                ...                  , 'k:')
                >>> _ = plt.plot(v - 1, (v - 1)**2/2, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'minimized $(N^3/3\kappa)\beta U$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        v = self.np_array(v)
        beta_U = np.zeros(len(v))
        p = np.zeros(len(v))
        s = np.zeros((self.N, len(v)))
        for i, v_i in enumerate(v):
            s[:, i] = la.inv(self.H_U_00()/self.kappa).dot(
                np.concatenate(([2*v_i], [-v_i], np.zeros(self.N - 4),
                                [-lambda_[0], 4*lambda_[0] - lambda_[1]])))
            p[i] = self.kappa*(v_i - (2*s[0, i] - s[1, i]))
            beta_U[i] = self.beta_U_00(v_i, s[:, i])
        return beta_U, p, s

    def minimize_beta_U(self, v, transition_state=False):
        r"""Function to minimize the potential energy of the system.

        Args:
            v (array_like): The nondimensional end separation.
            transition_state (bool, optional, default=False):
                Whether or not to fix the crack tip bond
                stretch at the transition state stretch.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional positions.

        Example:
            Plot the rescaled minimized nondimensional potential energy
            as a function of the nondimensional end separation
            for the mechnically-treated system
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackMechanical
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25]:
                ...     model = CrackMechanical(N=N)
                ...     beta_U = model.minimize_beta_U(v)[0]
                ...     _ = plt.plot(v - 1, model.N**3/3/model.kappa*beta_U,
                ...                  label='$N=$'+str(N))
                ...     p_m = model.p_mechanical(v)
                ...     beta_Pi = model.minimize_beta_Pi(p_m)[0]
                ...     _ = plt.plot(v - 1,
                ...                  model.N**3/3/model.kappa*(beta_Pi+p_m*v)
                ...                  , 'k:')
                >>> _ = plt.plot(v - 1, (v - 1)**2/2, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'minimized $(N^3/3\kappa)\beta U$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        v = self.np_array(v)
        beta_U = np.zeros(len(v))
        p = np.zeros(len(v))
        s = np.zeros((self.L, len(v)))
        for i, v_i in enumerate(v):
            s_guess = np.concatenate((
                self.minimize_beta_U_00(
                    v_i, [1 + transition_state*(self.lambda_TS - 1), 1]
                )[2][:, 0],
                np.ones(self.M - transition_state)
            ))
            if transition_state is True:
                res = minimize(
                    lambda s: self.beta_U(v_i, np.concatenate((
                        s[:self.N], [self.lambda_TS], s[-(self.M - 1):]
                    ))),
                    s_guess,
                    method='Newton-CG',
                    jac=lambda s: self.j_U_TS(v_i, s),
                    hess=self.H_U_TS
                )
                s[:self.N, i] = res.x[:self.N]
                s[self.N, i] = self.lambda_TS
                s[-(self.M - 1):, i] = res.x[-(self.M - 1):]
            else:
                res = minimize(
                    lambda s: self.beta_U(v_i, s),
                    s_guess,
                    method='Newton-CG',
                    jac=lambda s: self.j_U(v_i, s),
                    hess=self.H_U
                )
                s[:, i] = res.x
            p[i] = self.kappa*(v_i - (2*s[0, i] - s[1, i]))
            beta_U[i] = res.fun
        return beta_U, p, s

    def minimize_beta_Pi_00(self, p, lambda_):
        r"""Function to minimize the total potential energy
        of the isolated bending system.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional total potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional end separation.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional positions.

        """
        p = self.np_array(p)
        beta_Pi = np.zeros(len(p))
        v = np.zeros(len(p))
        s = np.zeros((self.N, len(p)))
        for i, p_i in enumerate(p):
            vs = la.inv(self.H_Pi_00()/self.kappa).dot(
                np.concatenate(([p_i/self.kappa], np.zeros(self.N - 2),
                                [-lambda_[0], 4*lambda_[0] - lambda_[1]])))
            v[i] = vs[0]
            s[:, i] = vs[1:]
            beta_Pi[i] = self.beta_Pi_00(p_i, v[i], s[:, i])
        return beta_Pi, v, s

    def minimize_beta_Pi(self, p, transition_state=False):
        """Function to minimize the total potential energy of the system.

        Args:
            p (array_like): The nondimensional end force.
            transition_state (bool, optional, default=False):
                Whether or not to fix the crack tip bond
                stretch at the transition state stretch.

        Returns:
            tuple:

                - (*numpy.ndarray*) -
                  The minimized nondimensional total potential energy.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional end separation.
                - (*numpy.ndarray*) -
                  The corresponding nondimensional positions.

        """
        p = self.np_array(p)
        beta_Pi = np.zeros(len(p))
        v = np.zeros(len(p))
        s = np.zeros((self.L, len(p)))
        for i, p_i in enumerate(p):
            v_guess, s_guess_0 = self.minimize_beta_Pi_00(
                p_i, [1 + transition_state*(self.lambda_TS - 1), 1]
            )[1:]
            s_guess = np.concatenate((
                v_guess,
                s_guess_0[:, 0],
                np.ones(self.M - transition_state)
            ))
            if transition_state is True:
                res = minimize(
                    lambda vs: self.beta_Pi(p_i, vs[0], np.concatenate((
                        vs[1:self.N + 1], [self.lambda_TS], vs[-(self.M - 1):]
                    ))),
                    s_guess,
                    method='Newton-CG',
                    jac=lambda vs: self.j_Pi_TS(p_i, vs[0], vs[1:]),
                    hess=lambda vs: self.H_Pi_TS(vs[1:])
                )
                s[:self.N, i] = res.x[1:self.N + 1]
                s[self.N, i] = self.lambda_TS
                s[-(self.M - 1):, i] = res.x[-(self.M - 1):]
            else:
                res = minimize(
                    lambda vs: self.beta_Pi(p_i, vs[0], vs[1:]),
                    s_guess,
                    method='Newton-CG',
                    jac=lambda vs: self.j_Pi(p_i, vs[0], vs[1:]),
                    hess=lambda vs: self.H_Pi(vs[1:])
                )
                s[:, i] = res.x[1:]
            v[i] = res.x[0]
            beta_Pi[i] = res.fun
        return beta_Pi, v, s

    def v_mechanical(self, p):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        for the mechnically-treated system,
        calculated by minimizing the total potential energy.

        Args:
            p (array_like): The nondimensional end force.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        Example:
            Plot the nondimensional end separation as a function of the
            rescaled nondimensional end force
            for the mechnically-treated system
            for an increasing nondimensional bond energy :math:`\varepsilon`
            and compare to the limit given by
            the mechanically-treated reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackMechanical
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [50, 100, 500]:
                ...     model = CrackMechanical(varepsilon=varepsilon)
                ...     r_p = np.linspace(0, 10, 33)
                ...     p = 3*model.kappa/model.N**3*r_p
                ...     v_m = model.v_mechanical(p)
                ...     _ = plt.plot(v_m - 1, r_p,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                ...     p_m = model.p_mechanical(v_m)
                ...     r_p_m = model.N**3/3/model.kappa*p_m
                ...     _ = plt.plot(v_m - 1, r_p_m, 'k:')
                >>> _ = plt.plot(model.v_0_mechanical(p, [1, 1]) - 1,
                ...              r_p, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v_m$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.minimize_beta_Pi(p)[1]

    def v_0_mechanical(self, p, lambda_):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        for the mechnically-treated reference system,
        calculated by minimizing the total potential energy.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        return self.v_b_mechanical(p, lambda_)

    def v_b_mechanical(self, p, lambda_):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        for the mechnically-treated isolated bending system,
        calculated by minimizing the total potential energy.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        return self.minimize_beta_Pi_00(p, lambda_)[1]

    def p_mechanical(self, v):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        for the mechnically-treated system,
        calculated by minimizing the potential energy.

        Args:
            v (array_like): The nondimensional end separation.

        Returns:
            numpy.ndarray: The nondimensional end force.

        Example:
            Plot the rescaled nondimensional end force
            as a function of the nondimensional end separation
            for the mechnically-treated system
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackMechanical
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25]:
                ...     model = CrackMechanical(N=N)
                ...     p_m = model.p_mechanical(v)
                ...     r_p_m = model.N**3/3/model.kappa*p_m
                ...     _ = plt.plot(v - 1, r_p_m, label='$N=$'+str(N))
                ...     v_m = model.v_mechanical(p_m)
                ...     _ = plt.plot(v_m - 1, r_p_m, 'k:')
                >>> _ = plt.plot(v - 1, v - 1, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p_m$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.minimize_beta_U(v)[1]

    def p_0_mechanical(self, v, lambda_):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        for the mechnically-treated reference system,
        calculated by minimizing the potential energy.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional end force.

        Example:
            Plot the rescaled nondimensional end force
            as a function of the nondimensional end separation
            for the mechnically-treated reference system
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackMechanical
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25]:
                ...     model = CrackMechanical(N=N)
                ...     p_m = model.p_0_mechanical(v, [1, 1])
                ...     r_p_m = model.N**3/3/model.kappa*p_m
                ...     _ = plt.plot(v - 1, r_p_m, label='$N=$'+str(N))
                ...     v_m = model.v_0_mechanical(p_m, [1, 1])
                ...     _ = plt.plot(v_m - 1, r_p_m, 'k:')
                >>> _ = plt.plot(v - 1, v - 1, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p_m$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.p_b_mechanical(v, lambda_)

    def p_b_mechanical(self, v, lambda_):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        for the mechnically-treated reference system,
        calculated by minimizing the potential energy.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional end force.

        """
        return self.minimize_beta_U_00(v, lambda_)[1]
