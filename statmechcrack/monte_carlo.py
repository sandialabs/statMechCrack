"""A module for the Metropolis-Hastings Markov chain Monte Carlo method.

This module consist of the class :class:`CrackMonteCarlo` which contains
methods to compute crack model quantities using the
Metropolis-Hastings Markov chain Monte Carlo method :cite:`haile1992molecular`.

"""

from sys import platform
import numpy as np
import numpy.linalg as la

from .mechanical import CrackMechanical

if platform in ('linux', 'linux2'):
    import multiprocessing as mp


class CrackMonteCarlo(CrackMechanical):
    """The Metropolis-Hastings Markov chain Monte Carlo class.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackMonteCarlo` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackMechanical` class instance.

        """
        CrackMechanical.__init__(self, **kwargs)
        self.beta_E = None

    def trial_config(self, prev_config, cov_config=1e-2):
        """Generates trial configurations given a previous configuration.

        Args:
            prev_config (numpy.ndarray): The previous configuration.
            cov_config (float, optional, default=1e-2): The covariance.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            numpy.ndarray: The generated trial configuration.

        """
        return prev_config + np.random.normal(0, cov_config, len(prev_config))

    def mh_next_config(self, prev_config, urng=np.random.uniform, **kwargs):
        """Generates accepted configurations given a previous configuration.

        This function returns an accepted configuration given a
        previous configuration. Trial configurations are generated and
        repeatedly rejected until accepted as the next configuration
        based on the Metropolis-Hastings algorithm.

        Args:
            prev_config (numpy.ndarray): The previous configuration.
            urng (object, optional, default=np.random.uniform):
                The uniform random number generator.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`trial_config`.

        Returns:
            numpy.ndarray: The accepted (next) configuration.

        """
        next_config = None
        while next_config is None:
            trial_config = self.trial_config(prev_config)
            delta_beta_E = self.beta_E(trial_config) - self.beta_E(prev_config)
            if (delta_beta_E < 0) or (np.exp(-delta_beta_E) > urng()):
                next_config = trial_config
        return next_config

    def mh_next_config_append(self, prev_configs, **kwargs):
        """Append next configuration onto a history of previous configurations.

        This method calls :meth:`mh_next_config` and appends the resulting next
        configuration onto a given history of previous configurations.

        Args:
            prev_configs (numpy.ndarray): The previous
                history of configurations.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            numpy.ndarray: The updated history of configurations.

        """
        return np.append(prev_configs, np.array(
            [self.mh_next_config(prev_configs[-1], **kwargs)]), axis=0)

    def burn_in(self, init_config, num_burns=10000, tol=np.inf, **kwargs):
        """Burn-in (throw away samples) until in the effective sampling region.

        This method runs a Monte Carlo calculation without utilizing samples,
        instead throwing them away until a desired number have been thrown away
        or a desired tolerance has been reached.
        This is typically done to obtain a burned-in configuration to use as
        the initial configuration for an actual Monte Carlo calculation.

        Args:
            init_config (numpy.ndarray):
                The initial configuration.
            num_burns (int, optional, default=10000):
                The number of samples to burn.
            tol (float, optional, default=np.inf):
                The burn-in tolerance.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config_append`.

        Returns:
            numpy.ndarray: The final, burned-in configuration.

        """
        config_history = np.array([init_config])
        prev_rmc = init_config
        n_res = tol + 1
        while n_res > tol or len(config_history) < num_burns:
            config_history = self.mh_next_config_append(
                config_history, **kwargs
            )
            next_rmc = np.mean(config_history, axis=0)
            n_res = la.norm(next_rmc - prev_rmc)/la.norm(prev_rmc)
            prev_rmc = next_rmc
        return config_history[-1]

    def parallel_calculation(self, serial_fun, init_config, **kwargs):
        """Monte Carlo calculation averaged over several parallel processes.
        This method performs several Monte Carlo calculations in parallel
        and returns the average result from all processes.
        Each Monte Carlo calculation is performed using ``serial_fun()``.
        The default is to utilize all available processors.

        Args:
            serial_fun (function):
                The function for a single Monte Carlo calculation.
            init_config (np.ndarray):
                The initial configuration for the burn-in process.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`burn_in` and ``serial_fun()``.

        Returns:
            float: The process-averaged result.

        Note:
            To avoid issues with ``multiprocessing`` and MacOS/Windows,
            ``num_processes`` is set to 1 for anything but Linux.

        """
        burned_in_config = self.burn_in(init_config, **kwargs)
        num_processes = 1
        if platform in ('linux', 'linux2'):
            num_processes = kwargs.get('num_processes', mp.cpu_count())
        if num_processes > 1:
            output = mp.Queue()

            def fun(seed, output):
                output.put(
                    serial_fun(
                        burned_in_config,
                        urng=np.random.RandomState(seed).random, **kwargs
                    )
                )

            processes = [
                mp.Process(target=fun, args=(seed, output))
                for seed in np.random.randint(88, size=num_processes)
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            process_results = [output.get() for p in processes]
            return np.mean(process_results)
        return serial_fun(burned_in_config, **kwargs)

    def p_isometric_monte_carlo_serial(self, v, init_config,
                                       num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`p_isometric_monte_carlo`.

        Args:
            v (array_like):
                The nondimensional end separation.
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            float: The nondimensional end force.

        """
        fun = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            fun_next = self.p_0_isometric(v, config)*np.exp(
                -self.beta_A_b_isometric(v, config)
            )
            fun += (fun_next - fun)/counter
            config = self.mh_next_config(config, **kwargs)
        return fun

    def p_isometric_monte_carlo(self, v, **kwargs):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            p(v) =
            e^{\beta\Delta A} \left\langle
                p_0 \, e^{-\beta\Delta A_0}
            \right\rangle_\star.

        Args:
            v (array_like): The nondimensional end separation.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            float: The nondimensional end force.

        """
        v = self.np_array(v)
        ensemble_average_fun = np.zeros(v.shape)
        for i, v_i in enumerate(v):
            self.beta_E = lambda lambda_: self.beta_U_1(lambda_) + \
                self.beta_A_0_abs_isometric(1, lambda_)

            def serial_fun(init_config, **kwargs):
                return self.p_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            ensemble_average_fun[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(v_i)[2][-self.M:, 0],
                **kwargs
            )
        beta_Delta_A = self.beta_A_isometric_monte_carlo(v, **kwargs)
        return np.exp(beta_Delta_A)*ensemble_average_fun

    def v_isotensional_monte_carlo_serial(self, p, init_config,
                                          num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`v_isotensional_monte_carlo`.

        Args:
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            float: The nondimensional end separation.

        """
        fun = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            fun_next = self.v_0_isotensional(p, config)*np.exp(
                -self.beta_G_b_isotensional(p, config)
            )
            fun += (fun_next - fun)/counter
            config = self.mh_next_config(config, **kwargs)
        return fun

    def v_isotensional_monte_carlo(self, p, **kwargs):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        in the isotensional ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            v(p) =
            e^{\beta\Delta G} \left\langle
                v_0 \, e^{-\beta\Delta G_0}
            \right\rangle_\star.

        Args:
            p (array_like): The nondimensional end force.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            float: The nondimensional end separation.

        """
        p = self.np_array(p)
        ensemble_average_fun = np.zeros(p.shape)
        for i, p_i in enumerate(p):
            self.beta_E = lambda lambda_: self.beta_U_1(lambda_) + \
                self.beta_G_0_abs_isotensional(0, lambda_)

            def serial_fun(init_config, **kwargs):
                return self.v_isotensional_monte_carlo_serial(
                        p_i, init_config, **kwargs
                    )

            ensemble_average_fun[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_Pi(p_i)[2][-self.M:, 0],
                **kwargs
            )
        beta_Delta_G = self.beta_G_isotensional_monte_carlo(p, **kwargs)
        return np.exp(beta_Delta_G)*ensemble_average_fun

    def beta_A_isometric_monte_carlo_serial(self, v, init_config,
                                            num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`beta_A_isometric_monte_carlo`.

        Args:
            v (array_like):
                The nondimensional end separation.
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            numpy.ndarray: The relative nondimensional Helmholtz free energy.

        """
        exp_neg_beta_Delta_A = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            exp_neg_beta_Delta_A_next = np.exp(
                -self.beta_A_b_isometric(v, config)
            )
            exp_neg_beta_Delta_A += (
                exp_neg_beta_Delta_A_next - exp_neg_beta_Delta_A
            )/counter
            config = self.mh_next_config(config, **kwargs)
        return np.log(1/exp_neg_beta_Delta_A)

    def beta_A_isometric_monte_carlo(self, v, **kwargs):
        r"""The relative nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            \beta\Delta A(v) =
            -\ln\left\langle e^{-\beta\Delta A_0} \right\rangle_\star,

        where the relative free energy of the reference system is

        .. math::
            \Delta A_0\equiv
            A_0(v,\boldsymbol{\lambda})-A_0(1,\boldsymbol{\lambda}),

        and where the isometric ensemble average here is

        .. math::
            \left\langle \phi \right\rangle_\star \equiv
            \frac{
                \int d\lambda \ e^{-\beta A_\star(\boldsymbol{\lambda})}
                    \, \phi(\boldsymbol{\lambda})
            }{
                \int d\lambda \ e^{-\beta A_\star(\boldsymbol{\lambda})}
            }
            ,

        where the free energy measure here is

        .. math::
            A_\star(\boldsymbol{\lambda})\equiv
            A_0(1,\boldsymbol{\lambda}) + U_1(\boldsymbol{\lambda}).

        Args:
            v (array_like): The nondimensional end separation.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            numpy.ndarray: The relative nondimensional Helmholtz free energy.

        """
        v = self.np_array(v)
        beta_A = np.zeros(v.shape)
        for i, v_i in enumerate(v):
            self.beta_E = lambda lambda_: self.beta_U_1(lambda_) + \
                self.beta_A_0_abs_isometric(1, lambda_)

            def serial_fun(init_config, **kwargs):
                return self.beta_A_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            beta_A[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(v_i)[2][-self.M:, 0],
                **kwargs
            )
        return beta_A

    def beta_G_isotensional_monte_carlo_serial(self, p, init_config,
                                               num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`beta_G_isotensional_monte_carlo`.

        Args:
            p (array_like): The nondimensional end force.
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            numpy.ndarray: The relative nondimensional Gibbs free energy.

        """
        exp_neg_beta_Delta_G = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            exp_neg_beta_Delta_G_next = np.exp(
                -self.beta_G_b_isotensional(p, config)
            )
            exp_neg_beta_Delta_G += (
                exp_neg_beta_Delta_G_next - exp_neg_beta_Delta_G
            )/counter
            config = self.mh_next_config(config, **kwargs)
        return np.log(1/exp_neg_beta_Delta_G)

    def beta_G_isotensional_monte_carlo(self, p, **kwargs):
        r"""The relative nondimensional Gibbs free energy
        as a function of the nondimensional end force
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            \beta\Delta G(v) =
            -\ln\left\langle e^{-\beta\Delta G_0} \right\rangle_\star,

        where the relative free energy of the reference system is

        .. math::
            \Delta G_0\equiv
            G_0(p,\boldsymbol{\lambda})-G_0(0,\boldsymbol{\lambda}),

        and where the isotensional ensemble average here is

        .. math::
            \left\langle \phi \right\rangle_\star \equiv
            \frac{
                \int d\lambda \ e^{-\beta G_\star(\boldsymbol{\lambda})}
                    \, \phi(\boldsymbol{\lambda})
            }{
                \int d\lambda \ e^{-\beta G_\star(\boldsymbol{\lambda})}
            }
            ,

        where the free energy measure here is

        .. math::
            G_\star(\boldsymbol{\lambda})\equiv
            G_0(0,\boldsymbol{\lambda}) + U_1(\boldsymbol{\lambda}).

        Args:
            p (array_like): The nondimensional end force.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            numpy.ndarray: The relative nondimensional Gibbs free energy.

        """
        p = self.np_array(p)
        beta_G = np.zeros(p.shape)
        for i, p_i in enumerate(p):
            self.beta_E = lambda lambda_: self.beta_U_1(lambda_) + \
                self.beta_G_0_abs_isotensional(0, lambda_)

            def serial_fun(init_config, **kwargs):
                return self.beta_G_isotensional_monte_carlo_serial(
                        p_i, init_config, **kwargs
                    )

            beta_G[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_Pi(p_i)[2][-self.M:, 0],
                **kwargs
            )
        return beta_G

    def k_isometric_monte_carlo_serial(self, v, init_config,
                                       num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`k_isometric_monte_carlo`.

        Args:
            v (array_like): The nondimensional end separation.
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        exp_neg_beta_Delta_A_0 = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            if len(init_config) == self.M:
                lambda_ = config
            else:
                lambda_ = np.concatenate(([self.lambda_TS], config))
            exp_neg_beta_Delta_A_0_next = np.exp(
                -self.beta_A_b_isometric(v, lambda_)
            )
            exp_neg_beta_Delta_A_0 += (
                exp_neg_beta_Delta_A_0_next - exp_neg_beta_Delta_A_0
            )/counter
            config = self.mh_next_config(config, **kwargs)
        return exp_neg_beta_Delta_A_0

    def k_isometric_monte_carlo(self, v, **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            \frac{k(v)}{k(1)} =
                \frac{\left\langle
                    e^{-\beta\Delta A_0^\ddagger}
                \right\rangle_\star^\ddagger}{\left\langle
                    e^{-\beta\Delta A_0}
                \right\rangle_\star}.

        Args:
            v (array_like): The nondimensional end separation.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        v = self.np_array(v)
        ensemble_average_fun = np.zeros(v.shape)
        for i, v_i in enumerate(v):
            self.beta_E = lambda lambda_: self.beta_U_1(lambda_) + \
                self.beta_A_0_abs_isometric(1, lambda_)

            def serial_fun(init_config, **kwargs):
                return self.k_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            ensemble_average_fun[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(v_i)[2][-self.M:, 0],
                **kwargs
            )
        ensemble_average_fun_TS = np.zeros(v.shape)
        for i, v_i in enumerate(v):
            self.beta_E = lambda lambda_: \
                self.beta_U_1(
                    np.concatenate(([self.lambda_TS], lambda_))
                ) + self.beta_A_0_abs_isometric(
                    1, np.concatenate(([self.lambda_TS], lambda_))
                )

            def serial_fun(init_config, **kwargs):
                return self.k_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            ensemble_average_fun_TS[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(
                    v_i, transition_state=True
                )[2][-(self.M - 1):, 0],
                **kwargs
            )
        return ensemble_average_fun_TS/ensemble_average_fun

    def k_isotensional_monte_carlo_serial(self, p, init_config,
                                          num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`k_isotensional_monte_carlo`.

        Args:
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        exp_neg_beta_Delta_G_0 = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            if len(init_config) == self.M:
                lambda_ = config
            else:
                lambda_ = np.concatenate(([self.lambda_TS], config))
            exp_neg_beta_Delta_G_0_next = np.exp(
                -self.beta_G_b_isotensional(p, lambda_)
            )
            exp_neg_beta_Delta_G_0 += (
                exp_neg_beta_Delta_G_0_next - exp_neg_beta_Delta_G_0
            )/counter
            config = self.mh_next_config(config, **kwargs)
        return exp_neg_beta_Delta_G_0

    def k_isotensional_monte_carlo(self, p, **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        in the isotensional ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            \frac{k(p)}{k(0)} =
                \frac{\left\langle
                    e^{-\beta\Delta G_0^\ddagger}
                \right\rangle_\star^\ddagger}{\left\langle
                    e^{-\beta\Delta G_0}
                \right\rangle_\star}.

        Args:
            p (array_like): The nondimensional end force.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        p = self.np_array(p)
        ensemble_average_fun = np.zeros(p.shape)
        for i, p_i in enumerate(p):
            self.beta_E = lambda lambda_: self.beta_U_1(lambda_) + \
                self.beta_G_0_abs_isotensional(0, lambda_)

            def serial_fun(init_config, **kwargs):
                return self.k_isotensional_monte_carlo_serial(
                        p_i, init_config, **kwargs
                    )

            ensemble_average_fun[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_Pi(p_i)[2][-self.M:, 0],
                **kwargs
            )
        ensemble_average_fun_TS = np.zeros(p.shape)
        for i, p_i in enumerate(p):
            self.beta_E = lambda lambda_: \
                self.beta_U_1(
                    np.concatenate(([self.lambda_TS], lambda_))
                ) + self.beta_G_0_abs_isotensional(
                    0, np.concatenate(([self.lambda_TS], lambda_))
                )

            def serial_fun(init_config, **kwargs):
                return self.k_isotensional_monte_carlo_serial(
                        p_i, init_config, **kwargs
                    )

            ensemble_average_fun_TS[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_Pi(
                    p_i, transition_state=True
                )[2][-(self.M - 1):, 0],
                **kwargs
            )
        return ensemble_average_fun_TS/ensemble_average_fun
