"""The core module for the crack model.

This module consist of the class :class:`Crack`
which, upon instantiation,
becomes a crack model instance with methods for calculating
quantities in either thermodynamic ensemble.
These thermodynamic quantities are calculated using an
asymptotic approach :cite:`buche2021fundamental`,
:cite:`buche2022freely`, :cite:`buchegrutzikufjc2022`
or a Monte Carlo approach (see :class:`.CrackMonteCarlo`).
Basic mathematical capabilities are provided by
``numpy`` :cite:`numpy` and ``scipy`` :cite:`scipy`,
and ``matplotlib`` :cite:`matplotlib` is often used
for displaying results.

"""

from .isotensional import CrackIsotensional


class Crack(CrackIsotensional):
    r"""The crack model class.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`Crack` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackIsotensional` class instance.

        """
        CrackIsotensional.__init__(self, **kwargs)

    def beta_A(self, v, ensemble='isometric', approach='asymptotic',
               absolute=False, **kwargs):
        """The nondimensional Helmholtz free energy
        as a function of the nondimensional end separation.

        Args:
            v (array_like): The nondimensional end separation.
            ensemble (str, optional, default='isometric'):
                The thermodynamic ensemble. The `isotensional`
                ensemble uses the Legendre transformation method.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            absolute (bool, optional, default=False):
                Whether not to use the absolute free energy.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.beta_A_isometric` or
                :meth:`p` and :meth:`~.beta_G_isotensional`.

        Returns:
            numpy.ndarray: The nondimensional Helmholtz free energy.

        """
        if ensemble == 'isometric':
            if absolute is True:
                beta_A = self.beta_A_abs_isometric(v, approach=approach)
            else:
                beta_A = self.beta_A_isometric(v, approach=approach, **kwargs)
        elif ensemble == 'isotensional':
            p = self.p(v, ensemble='isotensional', approach=approach, **kwargs)
            if absolute is True:
                beta_A = p*v + \
                    self.beta_G_abs_isotensional(p, approach=approach)
            else:
                beta_A = p*v + \
                    self.beta_G_isotensional(p, approach=approach, **kwargs)
        return beta_A

    def beta_A_0(self, v, lambda_, ensemble='isometric', absolute=False):
        """The nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        for the reference system.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isometric'):
                The thermodynamic ensemble. The `isotensional`
                ensemble uses the Legendre transformation method.
            absolute (bool, optional, default=False):
                Whether not to use the absolute free energy.

        Returns:
            numpy.ndarray: The nondimensional Helmholtz free energy.

        """
        if ensemble == 'isometric':
            if absolute is True:
                beta_A_0 = self.beta_A_0_abs_isometric(v, lambda_)
            else:
                beta_A_0 = self.beta_A_0_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            p = self.p_0(v, lambda_, ensemble='isotensional')
            if absolute is True:
                beta_A_0 = self.beta_G_0_abs_isotensional(p, lambda_) + p*v
            else:
                beta_A_0 = self.beta_G_0_isotensional(p, lambda_) + p*v
        return beta_A_0

    def beta_A_b(self, v, lambda_, ensemble='isometric', absolute=False):
        """The nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        for the isolated bending system.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isometric'):
                The thermodynamic ensemble. The `isotensional`
                ensemble uses the Legendre transformation method.
            absolute (bool, optional, default=False):
                Whether not to use the absolute free energy.

        Returns:
            numpy.ndarray: The nondimensional Helmholtz free energy.

        """
        if ensemble == 'isometric':
            if absolute is True:
                beta_A_b = self.beta_A_b_isometric_abs(v, lambda_)
            else:
                beta_A_b = self.beta_A_b_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            p = self.p_b(v, lambda_, ensemble='isotensional')
            if absolute is True:
                beta_A_b = self.beta_G_b_abs_isotensional(p, lambda_) + p*v
            else:
                beta_A_b = self.beta_G_b_isotensional(p, lambda_) + p*v
        return beta_A_b

    def beta_G(self, p, ensemble='isotensional',
               approach='asymptotic', absolute=False, **kwargs):
        """The nondimensional Gibbs free energy
        as a function of the nondimensional end force.

        Args:
            p (array_like): The nondimensional end force.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble. The `isometric`
                ensemble uses the Legendre transformation method.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            absolute (bool, optional, default=False):
                Whether not to use the absolute free energy.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.beta_G_isotensional` or
                :meth:`v` and :meth:`~.beta_A_isometric`.

        Returns:
            numpy.ndarray: The nondimensional Gibbs free energy.

        """
        if ensemble == 'isometric':
            v = self.v(p, ensemble='isometric', approach=approach, **kwargs)
            if absolute is True:
                beta_G = -p*v + \
                    self.beta_A_abs_isometric(v, approach=approach)
            else:
                beta_G = -p*v + \
                    self.beta_A_isometric(v, approach=approach, **kwargs)
        elif ensemble == 'isotensional':
            if absolute is True:
                beta_G = \
                    self.beta_G_abs_isotensional(p, approach=approach)
            else:
                beta_G = \
                    self.beta_G_isotensional(p, approach=approach, **kwargs)
        return beta_G

    def beta_G_0(self, p, lambda_, ensemble='isotensional', absolute=False):
        """The nondimensional Gibbs free energy
        as a function of the nondimensional end force
        for the reference system.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble. The `isometric`
                ensemble uses the Legendre transformation method.
            absolute (bool, optional, default=False):
                Whether not to use the absolute free energy.

        Returns:
            numpy.ndarray: The nondimensional Gibbs free energy.

        """
        if ensemble == 'isometric':
            v = self.v_0(p, lambda_, ensemble='isometric')
            if absolute is True:
                beta_G_0 = self.beta_A_0_abs_isometric(v, lambda_) - p*v
            else:
                beta_G_0 = self.beta_A_0_isometric(v, lambda_) - p*v
        elif ensemble == 'isotensional':
            if absolute is True:
                beta_G_0 = self.beta_G_0_abs_isotensional(p, lambda_)
            else:
                beta_G_0 = self.beta_G_0_isotensional(p, lambda_)
        return beta_G_0

    def beta_G_b(self, p, lambda_, ensemble='isotensional', absolute=False):
        """The nondimensional Gibbs free energy
        as a function of the nondimensional end force
        for the isolated bending system.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble. The `isometric`
                ensemble uses the Legendre transformation method.
            absolute (bool, optional, default=False):
                Whether not to use the absolute free energy.

        Returns:
            numpy.ndarray: The nondimensional Gibbs free energy.

        """
        if ensemble == 'isometric':
            v = self.v_b(p, lambda_, ensemble='isometric')
            if absolute is True:
                beta_G_b = self.beta_A_b_isometric_abs(v, lambda_) - p*v
            else:
                beta_G_b = self.beta_A_b_isometric(v, lambda_) - p*v
        elif ensemble == 'isotensional':
            if absolute is True:
                beta_G_b = self.beta_G_b_abs_isotensional(p, lambda_)
            else:
                beta_G_b = self.beta_G_b_isotensional(p, lambda_)
        return beta_G_b

    def p(self, v, ensemble='isometric', approach='asymptotic', **kwargs):
        r"""The nondimensional end force
        as a function of the nondimensional end separation.

        Args:
            v (array_like): The nondimensional end separation.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.p_isometric` or :meth:`~.v_isotensional`.

        Returns:
            numpy.ndarray: The nondimensional end force.

        """
        if ensemble == 'isometric':
            p = self.p_isometric(v, approach=approach, **kwargs)
        elif ensemble == 'isotensional':
            p = self.inv_fun(
                lambda p:
                    self.v_isotensional(p, approach=approach, **kwargs), v
            )
        return p

    def p_0(self, v, lambda_, ensemble='isometric'):
        """The nondimensional end force
        as a function of the nondimensional end separation
        for the reference system.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isometric'):
                The thermodynamic ensemble.

        Returns:
            numpy.ndarray: The nondimensional end force.

        """
        if ensemble == 'isometric':
            p_0 = self.p_0_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            p_0 = self.inv_fun(
                lambda p: self.v_0_isotensional(p, lambda_), v
            )
        return p_0

    def p_b(self, v, lambda_, ensemble='isometric'):
        """The nondimensional end force
        as a function of the nondimensional end separation
        for the isolated bending system.

        Args:
            v (array_like): The nondimensional end separation.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isometric'):
                The thermodynamic ensemble.

        Returns:
            numpy.ndarray: The nondimensional end force.

        """
        if ensemble == 'isometric':
            p_b = self.p_b_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            p_b = self.inv_fun(
                lambda p: self.v_b_isotensional(p, lambda_), v
            )
        return p_b

    def v(self, p, ensemble='isotensional', approach='asymptotic', **kwargs):
        """The nondimensional end separation
        as a function of the nondimensional end force.

        Args:
            p (array_like): The nondimensional end force.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.v_isotensional` or :meth:`~.p_isometric`.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        if ensemble == 'isometric':
            v = self.inv_fun(
                lambda v:
                    self.p_isometric(v, approach=approach, **kwargs), p
            )
        elif ensemble == 'isotensional':
            v = self.v_isotensional(p, approach=approach, **kwargs)
        return v

    def v_0(self, p, lambda_, ensemble='isotensional'):
        """The nondimensional end separation
        as a function of the nondimensional end force
        for the reference system.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        if ensemble == 'isometric':
            v_0 = self.inv_fun(
                lambda v: self.p_0_isometric(v, lambda_), p
            )
        elif ensemble == 'isotensional':
            v_0 = self.v_0_isotensional(p, lambda_)
        return v_0

    def v_b(self, p, lambda_, ensemble='isotensional'):
        """The nondimensional end separation
        as a function of the nondimensional end force
        for the isolated bending system.

        Args:
            p (array_like): The nondimensional end force.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        if ensemble == 'isometric':
            v_b = self.inv_fun(
                lambda v: self.p_b_isometric(v, lambda_), p
            )
        elif ensemble == 'isotensional':
            v_b = self.v_b_isotensional(p, lambda_)
        return v_b

    def k(self, p_or_v, ensemble='isometric', approach='asymptotic', **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        or the nondimensional end separation.

        Args:
            p_or_v (array_like):
                The nondimensional end force or position. Assumed to be
                the end separation for the isometric ensemble and
                the end force for the isotensional ensemble.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.
            approach (str, optional, default='asymptotic'):
                The calculation approach.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`~.k_isometric` or :meth:`~.k_isotensional`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        if ensemble == 'isometric':
            k = self.k_isometric(p_or_v, approach=approach, **kwargs)
        elif ensemble == 'isotensional':
            k = self.k_isotensional(p_or_v, approach=approach, **kwargs)
        return k

    def k_0(self, p_or_v, lambda_, ensemble='isometric'):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        or the nondimensional end separation
        for the reference system.

        Args:
            p_or_v (array_like):
                The nondimensional end force or position. Assumed to be
                the end separation for the isometric ensemble and
                the end force for the isotensional ensemble.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        if ensemble == 'isometric':
            k_0 = self.k_0_isometric(p_or_v, lambda_)
        elif ensemble == 'isotensional':
            k_0 = self.k_0_isotensional(p_or_v, lambda_)
        return k_0

    def k_b(self, p_or_v, lambda_, ensemble='isometric'):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        or the nondimensional end separation
        for the isolated bending system.

        Args:
            p_or_v (array_like):
                The nondimensional end force or position. Assumed to be
                the end separation for the isometric ensemble and
                the end force for the isotensional ensemble.
            lambda_ (array_like): The intact bond stretches.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        if ensemble == 'isometric':
            k_b = self.k_b_isometric(p_or_v, lambda_)
        elif ensemble == 'isotensional':
            k_b = self.k_b_isotensional(p_or_v, lambda_)
        return k_b
