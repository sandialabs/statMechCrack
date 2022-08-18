"""A module for testing examples within docstrings.

This module checks all other modules in the package
and ensures that the examples within each docstring
are executed properly and obtain the expected output.

"""

import doctest
import unittest
from os import path
from glob import glob


class Examples(unittest.TestCase):
    """Class to test examples within docstrings.

    """
    def main(self):
        """Main function for module-level testing functionality.

        """
        self.test_docstring_examples()

    def test_docstring_examples(self):
        """Function to test examples within docstrings.

        """
        tests_dir = path.dirname(__file__)
        files = glob(path.join(tests_dir, './*.py'))
        files += glob(path.join(tests_dir, '../*.py'))
        for file in files:
            self.assertFalse(doctest.testfile(file, module_relative=False)[0])


if __name__ == '__main__':
    unittest.main()
