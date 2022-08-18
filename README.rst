#############
statMechCrack
#############

|build| |docs| |codecov| |coveralls| |pyversions| |pypi| |conda| |docker| |license| |zenodo|

A Python package for a statistical mechanical model for crack growth. 

************
Installation
************

The package can be installed using ``pip`` via the `Python Package Index <https://pypi.org/project/statmechcrack>`_ (PyPI),

::

    pip install statmechcrack

or using ``conda`` via the ``mrbuche`` channel on `Anaconda <https://anaconda.org/mrbuche/statmechcrack>`_,

::

    conda install --channel mrbuche statmechcrack
    
Alternatively, a branch can be directly installed using

::

    pip install git+https://github.com/sandialabs/statmechcrack.git@<branch-name>

or after cloning a branch and executing ``python setup.py install``.
There are also `Docker images <https://hub.docker.com/r/mrbuche/statmechcrack>`_ available for use.
In all of these cases, a valid installation can be tested by running

::

    python -m statmechcrack.tests

***********
Information
***********

- `Contributing <https://statmechcrack.readthedocs.io/en/latest/CONTRIBUTING.html>`__
- `Documentation <https://statmechcrack.readthedocs.io/en/latest>`__
- `Examples <https://statmechcrack.readthedocs.io/en/latest/statmechcrack.examples.html>`__
- `License <https://github.com/sandialabs/statmechcrack/blob/main/LICENSE>`__
- `Releases <https://github.com/sandialabs/statmechcrack/releases>`__
- `Repository <https://github.com/sandialabs/statmechcrack>`__
- `Tutorial <https://statmechcrack.readthedocs.io/en/latest/TUTORIAL.html>`__

********
Citation
********

\M. R. Buche and S. J. Grutzik, ``statMechCrack``: a Python package for a statistical mechanical model for crack growth, `Zenodo (2022) <https://doi.org/10.????/zenodo.???????>`_.

*********
Copyright
*********

Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

..
    Badges ========================================================================

.. |docs| image:: https://img.shields.io/readthedocs/statmechcrack?logo=readthedocs&label=Read%20the%20Docs
    :target: https://statmechcrack.readthedocs.io/en/latest/

.. |build| image:: https://img.shields.io/github/workflow/status/sandialabs/statmechcrack/main?label=GitHub&logo=github
    :target: https://github.com/sandialabs/statmechcrack

.. |coveralls| image:: https://img.shields.io/coveralls/github/sandialabs/statmechcrack?logo=coveralls&label=Coveralls
    :target: https://coveralls.io/github/sandialabs/statmechcrack?branch=main

.. |codecov| image:: https://img.shields.io/codecov/c/github/sandialabs/statmechcrack?label=Codecov&logo=codecov
    :target: https://codecov.io/gh/sandialabs/statmechcrack

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/statmechcrack.svg?logo=python&logoColor=FBE072&color=4B8BBE&label=Python
    :target: https://pypi.org/project/statmechcrack/

.. |pypi| image:: https://img.shields.io/pypi/v/statmechcrack?logo=pypi&logoColor=FBE072&label=PyPI&color=4B8BBE
    :target: https://pypi.org/project/statmechcrack/

.. |conda| image:: https://img.shields.io/conda/v/mrbuche/statmechcrack.svg?logo=anaconda&color=3EB049&label=Anaconda
    :target: https://anaconda.org/mrbuche/statmechcrack/

.. |docker| image:: https://img.shields.io/docker/v/mrbuche/statmechcrack?color=0db7ed&label=Docker%20Hub&logo=docker&logoColor=0db7ed
    :target: https://hub.docker.com/r/mrbuche/statmechcrack

.. |license| image:: https://img.shields.io/github/license/sandialabs/statmechcrack?label=License
    :target: https://github.com/sandialabs/statmechcrack/blob/main/LICENSE

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.????/zenodo.???????.svg
    :target: https://doi.org/10.????/zenodo.???????