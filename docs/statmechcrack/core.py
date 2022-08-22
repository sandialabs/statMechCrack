"""The core module for the crack model.

This module consist of the class :class:`Crack`
which, upon instantiation,
becomes a crack model instance with methods for computing
quantities in either thermodynamic ensemble.

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

    def beta_A(self, v, ensemble='isometric',
               approach='asymptotic', absolute=False):
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

        Returns:
            numpy.ndarray: The nondimensional Helmholtz free energy.

        """
        if ensemble == 'isometric':
            if absolute is True:
                return self.beta_A_abs_isometric(v, approach=approach)
            else:
                return self.beta_A_isometric(v, approach=approach)
        elif ensemble == 'isotensional':
            p = self.p(v, ensemble='isotensional', approach=approach)
            if absolute is True:
                return p*v + \
                    self.beta_G_abs_isotensional(p, approach=approach)
            else:
                return p*v + \
                    self.beta_G_isotensional(p, approach=approach)

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
                return self.beta_A_0_abs_isometric(v, lambda_)
            else:
                return self.beta_A_0_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            p = self.p_0(v, lambda_, ensemble='isotensional')
            if absolute is True:
                return self.beta_G_0_abs_isotensional(p, lambda_) + p*v
            else:
                return self.beta_G_0_isotensional(p, lambda_) + p*v

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
                return self.beta_A_b_isometric_abs(v, lambda_)
            else:
                return self.beta_A_b_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            p = self.p_b(v, lambda_, ensemble='isotensional')
            if absolute is True:
                return self.beta_G_b_abs_isotensional(p, lambda_) + p*v
            else:
                return self.beta_G_b_isotensional(p, lambda_) + p*v

    def beta_G(self, p, ensemble='isotensional',
               approach='asymptotic', absolute=False):
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

        Returns:
            numpy.ndarray: The nondimensional Gibbs free energy.

        """
        if ensemble == 'isometric':
            v = self.v(p, ensemble='isometric', approach=approach)
            if absolute is True:
                return self.beta_A_abs_isometric(v, approach=approach) - p*v
            else:
                return self.beta_A_isometric(v, approach=approach) - p*v
        elif ensemble == 'isotensional':
            if absolute is True:
                return self.beta_G_abs_isotensional(p, approach=approach)
            else:
                return self.beta_G_isotensional(p, approach=approach)

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
                return self.beta_A_0_abs_isometric(v, lambda_) - p*v
            else:
                return self.beta_A_0_isometric(v, lambda_) - p*v
        elif ensemble == 'isotensional':
            if absolute is True:
                return self.beta_G_0_abs_isotensional(p, lambda_)
            else:
                return self.beta_G_0_isotensional(p, lambda_)

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
                return self.beta_A_b_isometric_abs(v, lambda_) - p*v
            else:
                return self.beta_A_b_isometric(v, lambda_) - p*v
        elif ensemble == 'isotensional':
            if absolute is True:
                return self.beta_G_b_abs_isotensional(p, lambda_)
            else:
                return self.beta_G_b_isotensional(p, lambda_)

    def p(self, v, ensemble='isometric', approach='asymptotic'):
        r"""The nondimensional end force
        as a function of the nondimensional end separation.

        Args:
            v (array_like): The nondimensional end separation.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.
            approach (str, optional, default='asymptotic'):
                The calculation approach.

        Returns:
            numpy.ndarray: The nondimensional end force.

        """
        if ensemble == 'isometric':
            return self.p_isometric(v, approach=approach)
        elif ensemble == 'isotensional':
            return self.inv_fun(
                lambda p: self.v_isotensional(p, approach=approach), v
            )

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
            return self.p_0_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            return self.inv_fun(
                lambda p: self.v_0_isotensional(p, lambda_), v
            )

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
            return self.p_b_isometric(v, lambda_)
        elif ensemble == 'isotensional':
            return self.inv_fun(
                lambda p: self.v_b_isotensional(p, lambda_), v
            )

    def v(self, p, ensemble='isotensional', approach='asymptotic'):
        """The nondimensional end separation
        as a function of the nondimensional end force.

        Args:
            p (array_like): The nondimensional end force.
            ensemble (str, optional, default='isotensional'):
                The thermodynamic ensemble.
            approach (str, optional, default='asymptotic'):
                The calculation approach.

        Returns:
            numpy.ndarray: The nondimensional end separation.

        """
        if ensemble == 'isometric':
            return self.inv_fun(
                lambda v: self.p_isometric(v, approach=approach), p
            )
        elif ensemble == 'isotensional':
            return self.v_isotensional(p, approach=approach)

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
            return self.inv_fun(
                lambda v: self.p_0_isometric(v, lambda_), p
            )
        elif ensemble == 'isotensional':
            return self.v_0_isotensional(p, lambda_)

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
            return self.inv_fun(
                lambda v: self.p_b_isometric(v, lambda_), p
            )
        elif ensemble == 'isotensional':
            return self.v_b_isotensional(p, lambda_)

    def k(self, p_or_v, ensemble='isometric', approach='asymptotic'):
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

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        if ensemble == 'isometric':
            return self.k_isometric(p_or_v, approach=approach)
        elif ensemble == 'isotensional':
            return self.k_isotensional(p_or_v, approach=approach)

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
            return self.k_0_isometric(p_or_v, lambda_)
        elif ensemble == 'isotensional':
            return self.k_0_isotensional(p_or_v, lambda_)

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
            return self.k_b_isometric(p_or_v, lambda_)
        elif ensemble == 'isotensional':
            return self.k_b_isotensional(p_or_v, lambda_)
