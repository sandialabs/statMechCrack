"""A module for the quasi-two-dimensional crack model treated mechanically.

"""

from ..utility import BasicUtility


class CrackQ2DMechanical(BasicUtility):
    """The quasi-two-dimensional crack model class treated mechanically.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackQ2DMechanical` class.

        Initialize and inherit all attributes and methods
        from a :class:`BasicUtility` class instance.

        """
        BasicUtility.__init__(self)
