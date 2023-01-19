"""The core module for the quasi-two-dimensional crack model.

"""

from .isotensional import CrackQ2DIsotensional


class CrackQ2D(CrackQ2DIsotensional):
    r"""The quasi-two-dimensional crack model class.

    """
    def __init__(self, **kwargs):
        """Initializes the :class:`CrackQ2D` class.

        Initialize and inherit all attributes and methods
        from a :class:`.CrackQ2DIsotensional` class instance.

        """
        CrackQ2DIsotensional.__init__(self, **kwargs)
