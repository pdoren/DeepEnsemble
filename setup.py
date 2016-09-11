from distutils.core import setup
from setuptools import find_packages

setup(
    name='libml',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/pdoren/correntropy-and-ensembles-in-deep-learning',
    license='MIT',
    author='pdoren',
    author_email='pablo.saavedra@ug.uchile.cl',
    description='Library for working with Ensemble Models.',
    install_requires=['numpy'],
    extras_require={
        'testing': [
            'mock',
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],
    },
)
