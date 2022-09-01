"""A module for the crack model in the isometric ensemble.

This module consist of the class :class:`CrackIsometric` which
contains methods for computing quantities in the
isometric (constant displacement) thermodynamic ensemble.

"""

import numpy as np
import numpy.linalg as la

from .monte_carlo import CrackMonteCarlo


class CrackIsometric(CrackMonteCarlo):
    """The crack model class for the isometric ensemble.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackIsometric` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackMonteCarlo` class instance.

        """
        CrackMonteCarlo.__init__(self, **kwargs)
        rescaled_A = self.H_U_00()/self.kappa
        inv_rescaled_A = la.inv(rescaled_A)
        self.a_U_00 = np.array([
            inv_rescaled_A[0, 0], inv_rescaled_A[0, 1],
            inv_rescaled_A[1, 0], inv_rescaled_A[1, 1],
            inv_rescaled_A[0, -1], inv_rescaled_A[0, -2],
            inv_rescaled_A[1, -1], inv_rescaled_A[1, -2],
            inv_rescaled_A[-1, 0], inv_rescaled_A[-1, 1],
            inv_rescaled_A[-2, 0], inv_rescaled_A[-2, 1],
            inv_rescaled_A[-1, -1], inv_rescaled_A[-1, -2],
            inv_rescaled_A[-2, -1], inv_rescaled_A[-2, -2],
        ])
        self.det_H_U_00 = la.det(self.H_U_00())

    def Q_isometric(self, v, transition_state=False):
        r"""The nondimensional isometric partition function
        as a function of the nondimensional end separation,

        .. math::
            Q(v) = \int d\lambda \
                Q_0(v,\boldsymbol{\lambda})
                e^{-\beta U_1(\boldsymbol{\lambda})},

        approximated using the asymptotic relation

        .. math::
            Q(v) \sim
            Q_0(v, \hat{\boldsymbol{\lambda}})
            \prod_{j=1}^M \sqrt{\frac{2\pi}{\beta u''(\hat{\lambda}_j)}}
            \, e^{-\beta u(\hat{\lambda}_j)},

        which is valid for :math:`\varepsilon\gg 1`,
        where :math:`\hat{\boldsymbol{\lambda}}`
        is from minimizing :math:`\beta U`.

        Args:
            v (array_like): The nondimensional end separation.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            transition_state (bool, optional, default=False):
                Whether or not to calculate in the transition state.

        Returns:
            numpy.ndarray: The nondimensional isometric partition function.

        Example:
            Plot the rescaled nondimensional equilibrium radial distribution
            as a function of the nondimensional end separation
            in the isometric ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 250)
                >>> _ = plt.figure()
                >>> for varepsilon in [10, 25, 100, 1000]:
                ...     model = CrackIsometric(varepsilon=varepsilon)
                ...     r_Q = (model.Q_isometric(v)/model.Q_isometric(1)
                ...     )**(model.N**3/3/model.kappa)
                ...     _ = plt.plot(v - 1, (v - 1)**2*r_Q,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                >>> r_Q_0 = (model.Q_0_isometric(v, [1, 1])
                ...     / model.Q_0_isometric(1, [1, 1])
                ... )**(model.N**3/3/model.kappa)
                >>> _ = plt.plot(v - 1, (v - 1)**2*r_Q_0,
                ...              'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(\Delta v)^2$' +
                ...     r'$\left[\frac{Q(v)}{Q(1)}\right]^{N^3/3\kappa}$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        lambda_hat = self.minimize_beta_U(
            v, transition_state=transition_state
        )[2][-self.M:]
        Q_isometric = self.Q_0_isometric(v, lambda_hat)
        Q_isometric *= np.prod(
            np.sqrt(
                2*np.pi/self.beta_u_pp(lambda_hat[transition_state:])
            ), axis=0
        )
        Q_isometric *= np.exp(-np.sum(self.beta_u(lambda_hat), axis=0))
        return Q_isometric

    def beta_A_abs_isometric(self, v, approach='asymptotic',
                             transition_state=False):
        r"""The absolute nondimensional Helmholtz free energy
        as a function of the nondimensional end separation,

        .. math::
            \beta A(v) = -\ln Q(v).

        Args:
            v (array_like): The nondimensional end separation.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            transition_state (bool, optional, default=False):
                Whether or not to calculate in the transition state.

        Returns:
            numpy.ndarray: The absolute nondimensional Helmholtz free energy.

        """
        if approach == 'asymptotic':
            lambda_hat = self.minimize_beta_U(
                v, transition_state=transition_state
            )[2][-self.M:]
            beta_A_abs_isometric = \
                self.beta_A_0_abs_isometric(v, lambda_hat)
            beta_A_abs_isometric += 0.5*np.sum(
                np.log(
                    self.beta_u_pp(lambda_hat[transition_state:])/2/np.pi
                ), axis=0
            )
            beta_A_abs_isometric += np.sum(self.beta_u(lambda_hat), axis=0)
        elif approach == 'monte carlo':
            beta_A_abs_isometric = np.nan*v
        return beta_A_abs_isometric

    def beta_A_isometric(self, v, approach='asymptotic', **kwargs):
        r"""The relative nondimensional Helmholtz free energy
        as a function of the nondimensional end separation,

        .. math::
            \beta\Delta A(v) = \beta A(v) - \beta A(1).

        Args:
            v (array_like): The nondimensional end separation.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.beta_A_isometric_monte_carlo`.

        Returns:
            numpy.ndarray: The relative nondimensional Helmholtz free energy.

        Example:
            Plot the rescaled nondimensional relative Helmholtz free energy
            as a function of the nondimensional end separation
            in the isometric ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [10, 25, 100, 1000]:
                ...     model = CrackIsometric(varepsilon=varepsilon)
                ...     beta_A = \
                ...         model.beta_A_isometric(v, approach='asymptotic')
                ...     _ = plt.plot(v - 1, model.N**3/3/model.kappa*beta_A,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                >>> beta_A_0 = model.beta_A_0_isometric(v, [1, 1])
                >>> _ = plt.plot(v - 1, model.N**3/3/model.kappa*beta_A_0,
                ...              'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)\Delta\beta A$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        if approach == 'asymptotic':
            beta_A_isometric = \
                self.beta_A_abs_isometric(v, approach=approach) - \
                self.beta_A_abs_isometric(1, approach=approach)
        elif approach == 'monte carlo':
            beta_A_isometric = self.beta_A_isometric_monte_carlo(v, **kwargs)
        return beta_A_isometric

    def p_isometric(self, v, approach='asymptotic', **kwargs):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        in the isometric ensemble,

        .. math::
            p(v) = \frac{\partial}{\partial v}\,\beta A(v).

        Args:
            v (array_like): The nondimensional end separation.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.p_isometric_monte_carlo`.

        Returns:
            numpy.ndarray: The nondimensional end force.

        Example:
            Plot the rescaled nondimensional end force
            as a function of the nondimensional end separation
            in the isometric ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [10, 25, 100, 1000]:
                ...     model = CrackIsometric(varepsilon=varepsilon)
                ...     p = model.p_isometric(v, approach='asymptotic')
                ...     _ = plt.plot(v - 1, model.N**3/3/model.kappa*p,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                >>> p_0 = model.p_0_isometric(v, [1, 1])
                >>> _ = plt.plot(v - 1, model.N**3/3/model.kappa*p_0,
                ...              'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        if approach == 'asymptotic':
            s_hat = self.minimize_beta_U(v)[2]
            lambda_hat = s_hat[-self.M:]
            p_isometric = self.p_0_isometric(v, lambda_hat)
        elif approach == 'monte carlo':
            p_isometric = self.p_isometric_monte_carlo(v, **kwargs)
        return p_isometric

    def k_isometric(self, v, approach='asymptotic', **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end separation
        in the isometric ensemble.

        Args:
            v (array_like): The nondimensional end separation.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.k_isometric_monte_carlo`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        Example:
            Plot the nondimensional forward reaction rate coefficient
            as a function of the nondimensional end separation
            in the isometric ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [10, 25, 100, 1000]:
                ...     model = CrackIsometric(varepsilon=varepsilon)
                ...     _ = plt.semilogy(
                ...         v - 1, model.k_isometric(v),
                ...         label=r'$\varepsilon=$'+str(varepsilon))
                >>> _ = plt.semilogy(v - 1, model.k_0_isometric(v, [1, 1]),
                ...                  'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$k$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        if approach == 'asymptotic':
            k_isometric = (
                self.Q_isometric(v, transition_state=True)/self.Q_isometric(v)
            ) / (
                self.Q_isometric(1, transition_state=True)/self.Q_isometric(1)
            )
        elif approach == 'monte carlo':
            k_isometric = self.k_isometric_monte_carlo(v, **kwargs)
        return k_isometric

    def Q_0_isometric(self, v, lambda_):
        r"""The nondimensional isometric partition function
        as a function of the nondimensional end separation
        for the reference system,

        .. math::
            Q_0(v,\boldsymbol{\lambda}) =
            Q_b(v,\lambda_1,\lambda_2)
            e^{-\beta U_{01}(\boldsymbol{\lambda})}.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional isometric partition function.

        Example:
            Plot the rescaled nondimensional equilibrium radial distribution
            as a function of the nondimensional end separation
            for the reference system in the isometric ensemble
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 250)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25, 100]:
                ...     model = CrackIsometric(N=N)
                ...     r_Q_0 = (model.Q_0_isometric(v, [1, 1])
                ...         / model.Q_0_isometric(1, [1, 1])
                ...     )**(model.N**3/3/model.kappa)
                ...     _ = plt.plot(v - 1, (v - 1)**2*r_Q_0,
                ...                  label='$N=$'+str(N))
                >>> _ = plt.plot(v - 1, (v - 1)**2*np.exp(-(v - 1)**2/2),
                ...              'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(\Delta v)^2$' +
                ...     r'$\left[\frac{Q_0(v)}{Q_0(1)}\right]^{N^3/3\kappa}$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.Q_b_isometric(v, lambda_) * \
            np.exp(-self.beta_U_01(lambda_))

    def beta_A_0_abs_isometric(self, v, lambda_):
        r"""The absolute nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        for the reference system,

        .. math::
            \beta A_0(v,\boldsymbol{\lambda}) =
            -\ln Q_0(v,\boldsymbol{\lambda}) =
            \beta A_b(v,\lambda_1,\lambda_2)
            + \beta U_{01}(\boldsymbol{\lambda}).

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The absolute nondimensional Helmholtz free energy.

        """
        return self.beta_A_b_isometric_abs(v, lambda_) \
            + self.beta_U_01(lambda_)

    def beta_A_0_isometric(self, v, lambda_):
        r"""The relative nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        for the reference system,

        .. math::
            \beta\Delta A_0(v,\boldsymbol{\lambda}) =
            \beta A_0(v,\boldsymbol{\lambda}) -
            \beta A_0(v_0,\boldsymbol{\lambda}).

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The relative nondimensional Helmholtz free energy.

        Example:
            Plot the rescaled nondimensional relative Helmholtz free energy
            as a function of the nondimensional end separation
            for the reference system in the isometric ensemble
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25, 100]:
                ...     model = CrackIsometric(N=N)
                ...     beta_A_0 = model.beta_A_0_isometric(v, [1, 1])
                ...     _ = plt.plot(v - 1, N**3/3/model.kappa*beta_A_0,
                ...                  label='$N=$'+str(N))
                >>> _ = plt.plot(v - 1, (v - 1)**2/2, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)\beta\Delta A_0$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.beta_A_0_abs_isometric(v, lambda_) \
            - self.beta_A_0_abs_isometric(1, lambda_)

    def p_0_isometric(self, v, lambda_):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        for the reference system in the isometric ensemble,

        .. math::
            p_0(v,\lambda_1,\lambda_2) =
            \frac{\partial}{\partial v}\,\beta A_0(v,\lambda_1,\lambda_2) =
            p_b(v,\boldsymbol{\lambda}).

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional end force.

        Example:
            Plot the rescaled nondimensional end force
            as a function of the nondimensional end separation
            for the reference system in the isometric ensemble
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsometric
                >>> v = np.linspace(1, 11, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25, 100]:
                ...     model = CrackIsometric(N=N)
                ...     p_0 = model.p_0_isometric(v, [1, 1])
                ...     _ = plt.plot(v - 1, N**3/3/model.kappa*p_0,
                ...                  label='$N=$'+str(N))
                >>> _ = plt.plot(v - 1, v - 1, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p_0$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.p_b_isometric(v, lambda_)

    def k_0_isometric(self, v, lambda_):
        """The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end separation
        for the reference system in the isometric ensemble.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        return self.k_b_isometric(v, lambda_)

    def Q_b_isometric(self, v, lambda_):
        r"""The nondimensional isometric partition function
        as a function of the nondimensional end separation
        for the isolated bending system,

        .. math::
            Q_b(v,\lambda_1,\lambda_2) =
            \sqrt{\frac{(2\pi)^N}{\det\mathbf{H}}}
            \,e^{\frac{1}{2}\mathbf{g}^T\cdot\mathbf{H}^{-1}\cdot\mathbf{g}-f},

        where :math:`\mathbf{H}`,
        the nondimensional Hessian of the potential energy
        for the isolated bending system, and
        :math:`\mathbf{g}` and :math:`f` are

        .. math::
            \mathbf{g}(v,\lambda_1,\lambda_2) = \kappa\left(
                2v, -v, 0, \ldots, 0, -\lambda_1, 4\lambda_1 - \lambda_2
            \right)^T,

        .. math::
            f(v,\lambda_1,\lambda_2) = \frac{\kappa}{2}\left[v^2 + \lambda_1^2
                + \left(2\lambda_1 - \lambda_2\right)^2\right]
            .

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional isometric partition function.

        """
        return np.exp(-self.beta_A_b_isometric_abs(v, lambda_))

    def beta_A_b_isometric_abs(self, v, lambda_):
        r"""The absolute nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        for the isolated bending system,

        .. math::
            \beta A_b(v,\lambda_1,\lambda_2) =
            -\ln Q_b(v,\lambda_1,\lambda_2).

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The absolute nondimensional Helmholtz free energy.

        """
        rescaled_b_inv_A_b = self.np_array(v)**2*(
            4*self.a_U_00[0]
            - 2*(self.a_U_00[1] + self.a_U_00[2]) + self.a_U_00[3]
        ) + self.np_array(v)*(
            (4*lambda_[0] - lambda_[1])*(
                2*(self.a_U_00[4] + self.a_U_00[8])
                - (self.a_U_00[6] + self.a_U_00[9])
            ) + lambda_[0]*(
                self.a_U_00[7] + self.a_U_00[11]
                - 2*(self.a_U_00[5] + self.a_U_00[10])
            )
        ) + lambda_[0]**2*(
            self.a_U_00[15]
        ) + (4*lambda_[0] - lambda_[1])**2*(
            self.a_U_00[12]
        ) - lambda_[0]*(4*lambda_[0] - lambda_[1])*(
            self.a_U_00[13] + self.a_U_00[14]
        )
        return self.kappa/2*(
                self.np_array(v)**2
                + lambda_[0]**2 + (2*lambda_[0] - lambda_[1])**2
                - rescaled_b_inv_A_b
            ) + np.log(self.det_H_U_00) - self.N*np.log(2*np.pi)/2

    def beta_A_b_isometric(self, v, lambda_):
        r"""The relative nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        for the isolated bending system,

        .. math::
            \beta\Delta A_b(v,\lambda_1,\lambda_2) =
            \beta A_b(v,\lambda_1,\lambda_2) -
            \beta A_b(v_0,\lambda_1,\lambda_2).

        In the thermodynamic limit of a large number of broken bonds
        :math:`N`, this function has the asymptotic relation

        .. math::
            \beta\Delta A_b(v,\lambda_1,\lambda_2) \sim
            \frac{3\kappa}{2N^3}(v - v_0)^2 \quad\text{for }N\gg 1.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The relative nondimensional Helmholtz free energy.

        """
        return self.beta_A_b_isometric_abs(v, lambda_) \
            - self.beta_A_b_isometric_abs(1, lambda_)

    def p_b_isometric(self, v, lambda_):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        for the isolated bending system in the isometric ensemble,

        .. math::
            p_b(v,\lambda_1,\lambda_2) =
            \frac{\partial}{\partial v}\,\beta A_b(v,\lambda_1,\lambda_2).

        In the thermodynamic limit of a large number of broken bonds
        :math:`N`, this function has the asymptotic relation

        .. math::
            p_b(v,\lambda_1,\lambda_2) \sim
            \frac{3\kappa}{N^3}(v - v_0) \quad\text{for }N\gg 1.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional end force.

        """
        rescaled_dbdv_inv_A_b = self.np_array(v)*(
                4*self.a_U_00[0]
                - 2*(self.a_U_00[1] + self.a_U_00[2])
                + self.a_U_00[3]
            ) + lambda_[0]*(self.a_U_00[7] - 2*self.a_U_00[5]) \
            + (4*lambda_[0] - lambda_[1])*(2*self.a_U_00[4] - self.a_U_00[6])
        return self.kappa*(v - rescaled_dbdv_inv_A_b)

    def k_b_isometric(self, v, lambda_):
        """The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end separation
        for the isolated bending system in the isometric ensemble.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        return (
            self.Q_b_isometric(v, [self.lambda_TS, lambda_[1]]) /
            self.Q_b_isometric(v, lambda_)
        ) / (
            self.Q_b_isometric(1, [self.lambda_TS, lambda_[1]]) /
            self.Q_b_isometric(1, lambda_)
        )
