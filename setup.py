import re
from os.path import join
from distutils.core import setup
from setuptools import find_packages


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


def get_version():
    VERSIONFILE = join('statmechcrack', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"[0-9+.0-9+.0-9+]*[a-zA-Z0-9]*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))


setup(
    name='statmechcrack',
    version=get_version(),
    package_dir={'statmechcrack': 'statmechcrack'},
    packages=find_packages(),
    description='The Python package for \
        a statistical mechanical model for crack growth.',
    long_description=read("README.rst"),
    author='Michael R. Buche, Scott J. Grutzik',
    author_email='mrbuche@sandia.gov, sjgrutz@sandia.gov',
    url='https://sandialabs.github.io/statMechCrack',
    license='BSD-3-Clause',
    keywords=['fracture mechanics',
              'statistical mechanics',
              'thermodynamics'],
    install_requires=['matplotlib', 'numpy', 'scipy'],
    extras_require={
      'docs': ['docutils<0.18,>=0.14', 'ipykernel', 'ipython',
               'jinja2>=3.0', 'nbsphinx', 'pycodestyle',
               'sphinx', 'sphinx-copybutton',
               'sphinx-rtd-theme', 'sphinxcontrib-bibtex'],
      'testing': ['pycodestyle', 'pylint', 'pytest', 'pytest-cov'],
      'all': ['docutils<0.18,>=0.14', 'ipykernel', 'ipython', 'jinja2>=3.0',
              'nbsphinx', 'pycodestyle', 'pylint', 'pytest', 'pytest-cov',
              'sphinx', 'sphinx-copybutton',
              'sphinx-rtd-theme', 'sphinxcontrib-bibtex']
    },
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    project_urls={
      'Anaconda': 'https://anaconda.org/mrbuche/statmechcrack',
      'Documentation': 'https://statmechcrack.readthedocs.io/en/latest',
      'GitHub': 'https://github.com/sandialabs/statmechcrack',
      'Issues': 'https://github.com/sandialabs/statmechcrack/issues',
    },
)
