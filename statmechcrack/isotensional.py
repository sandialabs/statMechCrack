"""A module for the crack model in the isotensional ensemble.

This module consist of the class :class:`CrackIsotensional` which
contains methods for computing quantities in the
isotensional (constant force) thermodynamic ensemble.

"""

import numpy as np
import numpy.linalg as la

from .isometric import CrackIsometric


class CrackIsotensional(CrackIsometric):
    """The crack model class for the isotensional ensemble.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackIsotensional` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackIsometric` class instance.

        """
        CrackIsometric.__init__(self, **kwargs)
        inv_rescaled_H_Pi_00 = la.inv(self.H_Pi_00()/self.kappa)
        self.a_Pi_00 = np.array([
            inv_rescaled_H_Pi_00[0, 0],
            inv_rescaled_H_Pi_00[0, -1], inv_rescaled_H_Pi_00[0, -2],
            inv_rescaled_H_Pi_00[-1, 0], inv_rescaled_H_Pi_00[-2, 0],
            inv_rescaled_H_Pi_00[-1, -1], inv_rescaled_H_Pi_00[-1, -2],
            inv_rescaled_H_Pi_00[-2, -1], inv_rescaled_H_Pi_00[-2, -2],
        ])
        self.det_H_Pi_00 = la.det(self.H_Pi_00())

    def Z_isotensional(self, p, transition_state=False):
        r"""The nondimensional isotensional partition function
        as a function of the nondimensional end force,

        .. math::
            Z(p) = \int d\lambda \ Z_0(p,\boldsymbol{\lambda})
            e^{-\beta U_1(\boldsymbol{\lambda})},

        approximated using the asymptotic relation

        .. math::
            Z(p) \sim
            Z_0(p, \hat{\boldsymbol{\lambda}})
            \prod_{j=1}^M \sqrt{\frac{2\pi}{\beta u''(\hat{\lambda}_j)}}
            \, e^{-\beta u(\hat{\lambda}_j)},

        which is valid for :math:`\varepsilon\gg 1`,
        where :math:`\hat{\boldsymbol{\lambda}}`
        is from minimizing :math:`\beta\Pi`.

        Args:
            p (array_like): The nondimensional end force.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            transition_state (bool, optional, default=False):
                Whether or not to calculate in the transition state.

        Example:
            Plot the rescaled nondimensional equilibrium radial distribution
            as a function of the nondimensional end force
            in the isotensional ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 250)
                >>> _ = plt.figure()
                >>> for varepsilon in [50, 100, 1000]:
                ...     model = CrackIsotensional(varepsilon=varepsilon)
                ...     p = 3*model.kappa/model.N**3*rp
                ...     r_Z = (model.Z_isotensional(0)/model.Z_isotensional(p)
                ...     )**(model.N**3/3/model.kappa)
                ...     _ = plt.plot(rp, rp**2*r_Z,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                >>> r_Z_0 = (model.Z_0_isotensional(0, [1, 1])
                ...     / model.Z_0_isotensional(p, [1, 1])
                ... )**(model.N**3/3/model.kappa)
                >>> _ = plt.plot(rp, rp**2*r_Z_0, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)^2p^2$' +
                ...     r'$\left[\frac{Z_0(0)}{Z_0(p)}\right]^{N^3/3\kappa}$')
                >>> _ = plt.legend()
                >>> plt.show()

        Returns:
            numpy.ndarray: The nondimensional isotensional partition function.

        """
        lambda_hat = self.minimize_beta_Pi(
            p, transition_state=transition_state
        )[2][-self.M:]
        Z_isotensional = self.Z_0_isotensional(p, lambda_hat)
        Z_isotensional *= np.prod(
            np.sqrt(
                2*np.pi/self.beta_u_pp(lambda_hat[transition_state:])
            ), axis=0
        )
        Z_isotensional *= np.exp(-np.sum(self.beta_u(lambda_hat), axis=0))
        return Z_isotensional

    def beta_G_abs_isotensional(self, p, approach='asymptotic',
                                transition_state=False):
        r"""The absolute nondimensional Gibbs free energy
        as a function of the nondimensional end force,

        .. math::
            \beta G(p) = -\ln Z(p).

        Args:
            p (array_like): The nondimensional end force.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            transition_state (bool, optional, default=False):
                Whether or not to calculate in the transition state.

        Returns:
            numpy.ndarray: The absolute nondimensional Gibbs free energy.

        """
        if approach == 'asymptotic':
            lambda_hat = self.minimize_beta_Pi(
                p, transition_state=transition_state
            )[2][-self.M:]
            beta_G_abs_isotensional = \
                self.beta_G_0_abs_isotensional(p, lambda_hat)
            beta_G_abs_isotensional += 0.5*np.sum(
                np.log(
                    self.beta_u_pp(lambda_hat[transition_state:])/2/np.pi
                ), axis=0
            )
            beta_G_abs_isotensional += np.sum(self.beta_u(lambda_hat), axis=0)
        elif approach == 'monte carlo':
            beta_G_abs_isotensional = np.nan*p
        return beta_G_abs_isotensional

    def beta_G_isotensional(self, p, approach='asymptotic', **kwargs):
        r"""The relative nondimensional Gibbs free energy
        as a function of the nondimensional end force,

        .. math::
            \beta\Delta G(p) = \beta G(p) - \beta G(0).

        Args:
            p (array_like): The nondimensional end force.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.beta_G_isotensional_monte_carlo`.

        Returns:
            numpy.ndarray: The relative nondimensional Gibbs free energy.

        Example:
            Plot the rescaled nondimensional relative Gibbs free energy
            as a function of the rescaled nondimensional end force
            in the isotensional ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [50, 100, 1000]:
                ...     model = CrackIsotensional(varepsilon=varepsilon)
                ...     p = 3*model.kappa/model.N**3*rp
                ...     beta_G = model.beta_G_isotensional(
                ...         p, approach='asymptotic')
                ...     _ = plt.plot(rp, model.N**3/3/model.kappa*beta_G,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                >>> beta_G_0 = model.beta_G_0_isotensional(p, [1, 1])
                >>> _ = plt.plot(rp, model.N**3/3/model.kappa*beta_G_0,
                ...              'k--', label='limit')
                >>> _ = plt.xlabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)\Delta\beta G$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        if approach == 'asymptotic':
            G_isotensional = \
                self.beta_G_abs_isotensional(p, approach=approach) - \
                self.beta_G_abs_isotensional(0, approach=approach)
        elif approach == 'monte carlo':
            G_isotensional = self.beta_G_isotensional_monte_carlo(p, **kwargs)
        return G_isotensional

    def v_isotensional(self, p, approach='asymptotic', **kwargs):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        in the isotensional ensemble,

        .. math::
            v(p) = -\frac{\partial}{\partial p}\,\beta G(p).

        Args:
            p (array_like): The nondimensional end force.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.v_isotensional_monte_carlo`.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        Example:
            Plot the nondimensional end separation
            as a function of the rescaled nondimensional end force
            in the isotensional ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [50, 100, 1000]:
                ...     model = CrackIsotensional(varepsilon=varepsilon)
                ...     p = 3*model.kappa/model.N**3*rp
                ...     v = model.v_isotensional(p, approach='asymptotic')
                ...     _ = plt.plot(v - 1, rp,
                ...                  label=r'$\varepsilon=$'+str(varepsilon))
                >>> v_0 = model.v_0_isotensional(p, [1, 1])
                >>> _ = plt.plot(v_0 - 1, rp, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        if approach == 'asymptotic':
            lambda_hat = self.minimize_beta_Pi(p)[2][-self.M:]
            v_isotensional = self.v_0_isotensional(p, lambda_hat)
        elif approach == 'monte carlo':
            v_isotensional = self.v_isotensional_monte_carlo(p, **kwargs)
        return v_isotensional

    def k_isotensional(self, p, approach='asymptotic', **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        in the isotensional ensemble.

        Args:
            p (array_like): The nondimensional end force.
            approach (str, optional, default='asymptotic'):
                The calculation approach.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        Example:
            Plot the nondimensional forward reaction rate coefficient
            as a function of the nondimensional end force
            in the isotensional ensemble
            for an increasing nondimensional bond energy
            and compare to the limit given by the reference system:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 33)
                >>> _ = plt.figure()
                >>> for varepsilon in [50, 100, 1000]:
                ...     model = CrackIsotensional(varepsilon=varepsilon)
                ...     p = 3*model.kappa/model.N**3*rp
                ...     _ = plt.semilogy(
                ...         rp, model.k_isotensional(p),
                ...         label=r'$\varepsilon=$'+str(varepsilon))
                >>> _ = plt.semilogy(rp, model.k_0_isotensional(p, [1, 1]),
                ...                  'k--', label='limit')
                >>> _ = plt.xlabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.ylabel(r'$k$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        if approach == 'asymptotic':
            k_isotensional = (
                self.Z_isotensional(p, transition_state=True) /
                self.Z_isotensional(p)
            ) / (
                self.Z_isotensional(0, transition_state=True) /
                self.Z_isotensional(0)
            )
        elif approach == 'monte carlo':
            k_isotensional = self.k_isotensional_monte_carlo(p, **kwargs)
        return k_isotensional

    def Z_0_isotensional(self, p, lambda_):
        r"""The nondimensional isotensional partition function
        as a function of the nondimensional end force
        for the reference system,

        .. math::
            Z_0(p,\boldsymbol{\lambda}) =
            Z_b(p,\lambda_1,\lambda_2)
            e^{-\beta U_{01}(\boldsymbol{\lambda})}.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional isotensional partition function.

        Example:
            Plot the rescaled nondimensional equilibrium radial distribution
            as a function of the nondimensional end force
            for the reference system in the isotensional ensemble
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 250)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25, 100]:
                ...     model = CrackIsotensional(N=N)
                ...     p = 3*model.kappa/N**3*rp
                ...     r_Z_0 = (model.Z_0_isotensional(0, [1, 1])
                ...         / model.Z_0_isotensional(p, [1, 1])
                ...     )**(model.N**3/3/model.kappa)
                ...     _ = plt.plot(rp, rp**2*r_Z_0, label='$N=$'+str(N))
                >>> _ = plt.plot(rp, rp**2*np.exp(-rp*(rp/2 + 1)),
                ...              'k--', label='limit')
                >>> _ = plt.xlabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)^2p^2$' +
                ...     r'$\left[\frac{Z_0(0)}{Z_0(p)}\right]^{N^3/3\kappa}$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.Z_b_isotensional(p, lambda_) * \
            np.exp(-self.beta_U_01(lambda_))

    def beta_G_0_abs_isotensional(self, p, lambda_):
        r"""The absolute nondimensional Gibbs free energy
        as a function of the nondimensional end force
        for the reference system,

        .. math::
            \beta G_0(p,\boldsymbol{\lambda}) =
            -\ln Z_0(p,\boldsymbol{\lambda}) =
            \beta G_b(p,\lambda_1,\lambda_2)
            + \beta U_{01}(\boldsymbol{\lambda}).

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The absolute nondimensional Gibbs free energy.

        """
        return self.beta_G_b_abs_isotensional(p, lambda_) \
            + self.beta_U_01(lambda_)

    def beta_G_0_isotensional(self, p, lambda_):
        r"""The relative nondimensional Gibbs free energy
        as a function of the nondimensional end force
        for the reference system,

        .. math::
            \beta\Delta G_0(p,\boldsymbol{\lambda}) =
            \beta G_0(p,\boldsymbol{\lambda}) -
            \beta G_0(0,\boldsymbol{\lambda}).

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The relative nondimensional Gibbs free energy.

        Example:
            Plot the rescaled nondimensional relative Gibbs free energy
            as a function of the rescaled nondimensional end force
            for the reference system in the isotensional ensemble
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25, 100]:
                ...     model = CrackIsotensional(N=N)
                ...     p = 3*model.kappa/N**3*rp
                ...     beta_G_0 = \
                ...         model.beta_G_0_isotensional(p, [1, 1])
                ...     _ = plt.plot(rp, N**3/3/model.kappa*beta_G_0,
                ...                  label='$N=$'+str(N))
                >>> _ = plt.plot(rp, -rp*(rp/2 + 1), 'k--', label='limit')
                >>> _ = plt.xlabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)\beta\Delta G_0$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.beta_G_0_abs_isotensional(p, lambda_) \
            - self.beta_G_0_abs_isotensional(0, lambda_)

    def v_0_isotensional(self, p, lambda_):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        for the reference system in the isotensional ensemble,

        .. math::
            v_0(p,\lambda_1,\lambda_2) = -\frac{\partial}{\partial p}
            \,\beta G_0(p,\lambda_1,\lambda_2) =
            v_b(p,\lambda_1,\lambda_2).

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (list): The stretch of the first two intact bonds.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        Example:
            Plot the nondimensional end separation
            as a function of the rescaled nondimensional end force
            for the reference system in the isotensional ensemble
            for an increasing number of broken bonds :math:`N`
            and compare to the thermodynamic limit:

            .. plot::

                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statmechcrack import CrackIsotensional
                >>> rp = np.linspace(0, 10, 33)
                >>> _ = plt.figure()
                >>> for N in [5, 10, 25, 100]:
                ...     model = CrackIsotensional(N=N)
                ...     p = 3*model.kappa/N**3*rp
                ...     v_0 = model.v_0_isotensional(p, [1, 1])
                ...     _ = plt.plot(v_0 - 1, rp, label='$N=$'+str(N))
                >>> _ = plt.plot(rp, rp, 'k--', label='limit')
                >>> _ = plt.xlabel(r'$\Delta v_0$')
                >>> _ = plt.ylabel(r'$(N^3/3\kappa)p$')
                >>> _ = plt.legend()
                >>> plt.show()

        """
        return self.v_b_isotensional(p, lambda_)

    def k_0_isotensional(self, p, lambda_):
        """The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        for the reference system in the isotensional ensemble.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        return self.k_b_isotensional(p, lambda_)

    def Z_b_isotensional(self, p, lambda_):
        r"""The nondimensional isotensional partition function
        as a function of the nondimensional end separation
        for the isolated bending system,

        .. math::
            Z_b(p,\lambda_1,\lambda_2) =
            \sqrt{\frac{(2\pi)^{N+1}}{\det\mathbf{H}}}
            \,e^{\frac{1}{2}\mathbf{g}^T\cdot\mathbf{H}^{-1}\cdot\mathbf{g}-f},

        where :math:`\mathbf{H}`,
        the nondimensional Hessian of the total potential energy
        for the isolated bending system, and
        :math:`\mathbf{g}` and :math:`f` are

        .. math::
            \mathbf{g}(p,\lambda_1,\lambda_2) = \kappa\left(
                p/\kappa, 0, \ldots, 0, -\lambda_1, 4\lambda_1 - \lambda_2
            \right)^T,

        .. math::
            f(\lambda_1,\lambda_2) = \frac{\kappa}{2}\left[\lambda_1^2
                + \left(2\lambda_1 - \lambda_2\right)^2\right]
            .

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional isotensional partition function.

        """
        return np.exp(-self.beta_G_b_abs_isotensional(p, lambda_))

    def beta_G_b_abs_isotensional(self, p, lambda_):
        r"""The absolute nondimensional Gibbs free energy
        as a function of the nondimensional end force
        for the isolated bending system,

        .. math::
            \beta G_b(p,\lambda_1,\lambda_2) =
            -\ln Z_b(p,\lambda_1,\lambda_2).

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray:
                The absolute nondimensional Gibbs free energy.

        """
        rescaled_b_inv_A_b = self.np_array(p/self.kappa)**2*(
            self.a_Pi_00[0]
        ) + self.np_array(p/self.kappa)*(
            (4*lambda_[0] - lambda_[1])*(
                self.a_Pi_00[1] + self.a_Pi_00[3]
            ) - lambda_[0]*(
                self.a_Pi_00[2] + self.a_Pi_00[4]
            )
        ) + lambda_[0]**2*(
            self.a_Pi_00[8]
        ) + (4*lambda_[0] - lambda_[1])**2*(
            self.a_Pi_00[5]
        ) - lambda_[0]*(4*lambda_[0] - lambda_[1])*(
            self.a_Pi_00[6] + self.a_Pi_00[7]
        )
        return self.kappa/2*(
                + lambda_[0]**2 + (2*lambda_[0] - lambda_[1])**2
                - rescaled_b_inv_A_b
            ) + np.log(self.det_H_Pi_00) - (self.N + 1)*np.log(2*np.pi)/2

    def beta_G_b_isotensional(self, p, lambda_):
        r"""The relative nondimensional Gibbs free energy
        as a function of the nondimensional end force
        for the isolated bending system,

        .. math::
            \beta\Delta G_b(p,\lambda_1,\lambda_2) =
            \beta G_b(p,\lambda_1,\lambda_2) -
            \beta G_b(0,\lambda_1,\lambda_2).

        In the thermodynamic limit of a large number of broken bonds
        :math:`N`, this function has the asymptotic relation

        .. math::
            \beta\Delta G_b(p,\lambda_1,\lambda_2) \sim
            -\frac{N^3}{6\kappa}\,p^2 - p
            \quad\text{for }N\gg 1.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray:
                The relative nondimensional Gibbs free energy.

        """
        return self.beta_G_b_abs_isotensional(p, lambda_) - \
            self.beta_G_b_abs_isotensional(0, lambda_)

    def v_b_isotensional(self, p, lambda_):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        for the isolated bending system in the isotensional ensemble,

        .. math::
            v_b(p,\lambda_1,\lambda_2) = -\frac{\partial}{\partial p}
            \,\beta G_b(p,\lambda_1,\lambda_2).

        In the thermodynamic limit of a large number of broken bonds
        :math:`N`, this function has the asymptotic relation

        .. math::
            v_b(p,\lambda_1,\lambda_2) \sim
            v_0 + \frac{N^3}{3\kappa}\,p \quad\text{for }N\gg 1.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (list): The stretch of the first two intact bonds.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        return self.np_array(p/self.kappa)*(
            self.a_Pi_00[0]
        ) + 0.5*(
            (4*lambda_[0] - lambda_[1])*(
                self.a_Pi_00[1] + self.a_Pi_00[3]
            ) - lambda_[0]*(
                self.a_Pi_00[2] + self.a_Pi_00[4]
            )
        )

    def k_b_isotensional(self, p, lambda_):
        """The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        for the isolated bending system in the isotensional ensemble.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        return (
            self.Z_b_isotensional(p, [self.lambda_TS, lambda_[1]]) /
            self.Z_b_isotensional(p, lambda_)
        ) / (
            self.Z_b_isotensional(0, [self.lambda_TS, lambda_[1]]) /
            self.Z_b_isotensional(0, lambda_)
        )
