from distutils.core import setup
from setuptools import find_packages
import os
import re

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str in Python 2 and 3
    with open(os.path.join(here, 'deepensemble', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''

setup(
    name='deepensemble',
    version=version,
    packages=find_packages(),
    url='https://github.com/pdoren/DeepEnsemble',
    download_url='https://github.com/pdoren/DeepEnsemble/tarball/0.1',
    license='MIT',
    author='pdoren',
    author_email='pablo.saavedra@ug.uchile.cl',
    description='Library for working with Ensemble Models.',
    keywords="",
    include_package_data=False,
    zip_safe=False,
    install_requires=['numpy', 'theano', 'scikit-learn', 'matplotlib'],
    extras_require={
            'testing': [
            'mock',
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],
    },
)
