"""Initializes the package.

"""

from .core import Crack
from .isometric import CrackIsometric
from .isotensional import CrackIsotensional
from .mechanical import CrackMechanical
from .monte_carlo import CrackMonteCarlo
from .q2d.core import CrackQ2D
from .q2d.isometric import CrackQ2DIsometric
from .q2d.isotensional import CrackQ2DIsotensional
from .q2d.mechanical import CrackQ2DMechanical


__version__ = "0.7.0"
