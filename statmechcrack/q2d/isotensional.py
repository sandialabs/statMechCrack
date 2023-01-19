"""A module for the quasi-two-dimensional crack model
in the isotensional ensemble.

"""

from .isometric import CrackQ2DIsometric


class CrackQ2DIsotensional(CrackQ2DIsometric):
    """The quasi-two-dimensional crack model class
    for the isotensional ensemble.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackQ2DIsotensional` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackQ2DIsometric` class instance.

        """
        CrackQ2DIsometric.__init__(self, **kwargs)
