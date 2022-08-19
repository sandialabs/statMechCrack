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

    def trial_config(self, prev_config, cov_config=1e-2, **kwargs):
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
            trial_config = self.trial_config(prev_config, **kwargs)
            delta_beta_e = self.beta_e(trial_config) - self.beta_e(prev_config)
            if (delta_beta_e < 0) or (np.exp(-delta_beta_e) > urng()):
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
        p = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            p_next = self.kappa*(v - (2*config[0] - config[1]))
            p += (p_next - p)/counter
            config = self.mh_next_config(config, **kwargs)
        return p

    def p_isometric_monte_carlo(self, v, **kwargs):
        r"""The nondimensional end force
        as a function of the nondimensional end separation
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            p(v) = \kappa\langle v - 2s_1 + s_2\rangle.

        Args:
            v (array_like): The nondimensional end separation.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            float: The nondimensional end force.

        """
        v = self.np_array(v)
        p = np.zeros(v.shape)
        for i, v_i in enumerate(v):
            self.beta_e = lambda config: self.beta_U(v_i, config)

            def serial_fun(init_config, **kwargs):
                return self.p_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            p[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(v_i)[2][:, 0],
                **kwargs
            )
        return p

    def v_isotensional_monte_carlo_serial(self, init_config,
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
        v = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            v_next = config[0]
            v += (v_next - v)/counter
            config = self.mh_next_config(config, **kwargs)
        return v

    def v_isotensional_monte_carlo(self, p, **kwargs):
        r"""The nondimensional end separation
        as a function of the nondimensional end force
        in the isotensional ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            v(p) = \langle s_0\rangle.

        Args:
            p (array_like): The nondimensional end force.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            float: The nondimensional end separation.

        """
        p = self.np_array(p)
        v = np.zeros(p.shape)
        for i, p_i in enumerate(p):
            self.beta_e = lambda vs: self.beta_Pi(p_i, vs[0], vs[1:])
            v_guess, s_guess = self.minimize_beta_Pi(p_i)[1:]
            v[i] = self.parallel_calculation(
                self.v_isotensional_monte_carlo_serial,
                np.concatenate((v_guess, s_guess[:, 0])),
                **kwargs
            )
        return v

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
        exp_neg_beta_A = 0
        config = init_config
        for counter in range(1, 1 + num_samples):
            exp_neg_beta_A_next = np.exp(
                self.beta_U(1, config) - self.beta_U(v, config)
            )
            exp_neg_beta_A += (exp_neg_beta_A_next - exp_neg_beta_A)/counter
            config = self.mh_next_config(config, **kwargs)
        return np.log(1/exp_neg_beta_A)

    def beta_A_isometric_monte_carlo(self, v, **kwargs):
        r"""The relative nondimensional Helmholtz free energy
        as a function of the nondimensional end separation
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            \beta\Delta A(v) =
            -\ln\left\langle e^{-\beta\Delta U}\right\rangle_{v=1},

        where :math:`\Delta U \equiv U(v,s) - U(1,s)`.

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
            self.beta_e = lambda config: self.beta_U(1, config)

            def serial_fun(init_config, **kwargs):
                return self.beta_A_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            beta_A[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(v_i)[2][:, 0],
                **kwargs
            )
        return beta_A

    def beta_G_isotensional_monte_carlo_serial(self, init_config,
                                               num_samples=1000000, **kwargs):
        """Serial calculation for :meth:`beta_G_isotensional_monte_carlo`.

        Args:
            init_config (np.ndarray):
                The initial configuration, typically already burned-in.
            num_samples (int, optional, default=1000000):
                The number of samples to use.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`mh_next_config`.

        Returns:
            numpy.ndarray: The relative nondimensional Gibbs free energy.

        """
        pass

    def beta_G_isotensional_monte_carlo(self, p, **kwargs):
        r"""The relative nondimensional Gibbs free energy
        as a function of the nondimensional end force
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            \beta\Delta G(v) =
            -\ln\left\langle e^{-\beta\Delta\Pi}\right\rangle_{p=0},

        where :math:`\Delta\Pi \equiv \Pi(p,s) - \Pi(0,s)`.

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
            self.beta_e = lambda vs: self.beta_Pi(p_i, vs[0], vs[1:])
            v_guess, s_guess = self.minimize_beta_Pi(p_i)[1:]
            beta_G[i] = self.parallel_calculation(
                self.beta_G_isotensional_monte_carlo_serial,
                np.concatenate((v_guess, s_guess[:, 0])),
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
        pass

    def k_isometric_monte_carlo(self, v, **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        in the isometric ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            k = \frac{
                    \left\langle e^{-\beta U}\right\rangle_{v=1}^\ddagger
                }{
                    \left\langle e^{-\beta U}\right\rangle_{v=1}
                }.

        Args:
            v (array_like): The nondimensional end separation.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        v = self.np_array(v)
        k = np.zeros(v.shape)
        for i, v_i in enumerate(v):
            self.beta_e = lambda config: self.beta_U(v_i, config)

            def serial_fun(init_config, **kwargs):
                return self.k_isometric_monte_carlo_serial(
                        v_i, init_config, **kwargs
                    )

            k[i] = self.parallel_calculation(
                serial_fun,
                self.minimize_beta_U(v_i)[2][:, 0],
                **kwargs
            )
        return k

    def k_isotensional_monte_carlo_serial(self, init_config,
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
        pass

    def k_isotensional_monte_carlo(self, p, **kwargs):
        r"""The nondimensional forward reaction rate coefficient
        as a function of the nondimensional end force
        in the isotensional ensemble, using a
        Metropolis-Hastings Markov chain Monte Carlo calculation
        of the ensemble average

        .. math::
            k = \frac{
                    \left\langle e^{-\beta\Pi}\right\rangle_{p=0}^\ddagger
                }{
                    \left\langle e^{-\beta\Pi}\right\rangle_{p=0}
                }.

        Args:
            p (array_like): The nondimensional end force.
            **kwargs: Arbitrary keyword arguments.
                Passed to :meth:`parallel_calculation`.

        Returns:
            numpy.ndarray: The nondimensional forward reaction rate.

        """
        p = self.np_array(p)
        k = np.zeros(p.shape)
        for i, p_i in enumerate(p):
            self.beta_e = lambda vs: self.beta_Pi(p_i, vs[0], vs[1:])
            v_guess, s_guess = self.minimize_beta_Pi(p_i)[1:]
            k[i] = self.parallel_calculation(
                self.k_isotensional_monte_carlo_serial,
                np.concatenate((v_guess, s_guess[:, 0])),
                **kwargs
            )
        return k
