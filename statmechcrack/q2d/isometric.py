"""A module for the quasi-two-dimensional crack model
in the isometric ensemble.

"""

from .mechanical import CrackQ2DMechanical


class CrackQ2DIsometric(CrackQ2DMechanical):
    """The quasi-two-dimensional crack model class
    for the isometric ensemble.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackQ2DIsometric` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackQ2DMechanical` class instance.

        """
        CrackQ2DMechanical.__init__(self, **kwargs)
