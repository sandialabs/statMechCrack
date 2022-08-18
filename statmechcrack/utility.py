"""A module for basic crack model utilities.

This module consist of the class :class:`BasicUtility` which contains
attributes and methods that are meant to be inherited and utilized as
basic utilities by an arbitrary crack model class.

"""

import sys
import numpy as np
from scipy.optimize import root_scalar


class BasicUtility:
    """The crack model basic utilities class.

    This class contains
    attributes and methods that are meant to be inherited and utilized as
    basic utilities by an arbitrary crack model class.

    Attributes:
        minimum float:
            Helps avoid overflow when dividing by small numbers.
        maximum_float:
            Helps avoid overflow in function evaluations.
        maximum_exponent:
            Helps avoid overflow in exponential functions.

    """
    def __init__(self):
        """Initializes the :class:`BasicUtility` class.

        Inherits overflow-related parameters from ``sys.float_info``.

        """
        self.h = 1e-6
        self.minimum_float = sys.float_info.min
        self.maximum_float = sys.float_info.max
        self.maximum_exponent = np.log(self.maximum_float)

    def np_array(self, input_):
        """Function to return input as a numpy array.

        This function essentially serves as a wrapper for ``numpy.array``
        that returns non-array type inputs as shape (1,) numpy arrays
        rather than shape () numpy arrays, which is useful for indexing.

        Args:
            input_ (array_like): Anything passable to ``numpy.array``.

        Returns:
            numpy.ndarray: The input as an index-able numpy array.

        Example:
            Compare attempts to index the numpy array of an integer:

                >>> import numpy as np
                >>> from statmechcrack.utility import BasicUtility
                >>> try:
                ...     np.array(8)[0]
                ... except IndexError:
                ...     print('Error indexing 0-dimensional array.')
                ... finally:
                ...     BasicUtility().np_array(8)[0]
                Error indexing 0-dimensional array.
                8

        """
        if not np.array(input_).shape:
            return np.array([input_])
        return np.array(input_)

    def inv_fun(self, fun, val, guess=None, **kwargs):
        """Function to invert a mathematical function.

        This function returns the argument x given a function f(x)
        and y, the query value of the function, i.e. y = f(x).

        Note:
            This function is only meant for bijective f(x).

        Args:
            fun (function): The function f(x).
            val (array_like): The value(s) of the function y.
            guess (list, optional): A list of two initial guesses.
            **kwargs: Arbitrary keyword arguments.
                Passed to ``root_scalar``.

        Returns:
            numpy.ndarray: The corresponding argument(s) x.

        """
        vals = self.np_array(val)
        arg = np.zeros(vals.shape)
        for i, val_i in enumerate(vals):
            def res(arg):
                return fun(arg) - val_i
            if guess is not None:
                guess_0 = guess*0.95
                guess_1 = guess
            elif val_i == 0:
                guess_0 = 0
                guess_1 = 1e-2
            else:
                guess_0 = val_i*0.95
                guess_1 = val_i
            arg[i] = root_scalar(res, x0=guess_0, x1=guess_1, **kwargs).root
        return arg
