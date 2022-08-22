"""Module-level functionality for testing installation.

Example:
    Test that the package was installed properly:

    ::

        python -m statmechcrack.tests

    The example and style testing can also be included:

    ::

        python -m statmechcrack.tests -x

"""

import argparse

from .test_zero import Zero
from .test_one import One
from .test_many import Many
from .test_boundary import Boundary
from .test_interface import Interface

Zero().main()
One().main()
Many().main()
Boundary().main()
Interface().main()

parser = argparse.ArgumentParser()
parser.add_argument('-x', action='store_true')
args = parser.parse_args()
if parser.parse_args().x is True:
    from .test_examples import Examples
    from .test_style import Style
    Examples().main()
    Style().main()
