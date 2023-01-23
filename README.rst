#############
statMechCrack
#############

|build| |docs| |codefactor|

Statistical mechanical models for crack growth. 

************
Installation
************

|pypi| |conda|

The package can be installed using ``pip`` via the `Python Package Index <https://pypi.org/project/statmechcrack>`_ (PyPI),

::

    pip install statmechcrack

or using ``conda`` via the ``mrbuche`` channel on `Anaconda <https://anaconda.org/mrbuche/statmechcrack>`_,

::

    conda install --channel mrbuche statmechcrack
    
Alternatively, a branch can be directly installed using

::

    pip install git+https://github.com/sandialabs/statmechcrack.git@<branch-name>

or after cloning a branch and executing ``python setup.py install``. Any installation is tested by executing ``python -m statmechcrack.tests``.

********
Citation
********

|zenodo|

\M. R. Buche and S. J. Grutzik, ``statMechCrack``: statistical mechanical models for crack growth, `Zenodo (2023) <https://doi.org/10.5281/zenodo.7008312>`_.

*********
Copyright
*********

|license|

Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

..
    Badges ========================================================================

.. |build| image:: https://img.shields.io/github/actions/workflow/status/sandialabs/statmechcrack/main.yml?branch=main&label=GitHub&logo=github
    :target: https://github.com/sandialabs/statmechcrack

.. |docs| image:: https://img.shields.io/readthedocs/statmechcrack?logo=readthedocs&label=Read%20the%20Docs
    :target: https://statmechcrack.readthedocs.io/en/latest/

.. |codefactor| image:: https://img.shields.io/codefactor/grade/github/sandialabs/statmechcrack?label=Codefactor&logo=codefactor
   :target: https://www.codefactor.io/repository/github/sandialabs/statmechcrack

.. |pypi| image:: https://img.shields.io/pypi/v/statmechcrack?logo=pypi&logoColor=FBE072&label=PyPI&color=4B8BBE
    :target: https://pypi.org/project/statmechcrack/

.. |conda| image:: https://img.shields.io/conda/v/mrbuche/statmechcrack.svg?logo=anaconda&color=3EB049&label=Anaconda
    :target: https://anaconda.org/mrbuche/statmechcrack/

.. |license| image:: https://img.shields.io/github/license/sandialabs/statmechcrack?label=License&color=yellowgreen
    :target: https://github.com/sandialabs/statmechcrack/blob/main/LICENSE

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7008312.svg
    :target: https://doi.org/10.5281/zenodo.7008312
