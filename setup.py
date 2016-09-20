from distutils.core import setup
from setuptools import find_packages
import deepensemble

setup(
    name='deepensemble',
    version=deepensemble.__version__,
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
