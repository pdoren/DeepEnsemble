from distutils.core import setup

setup(
    name='correntropy-and-ensembles-in-deep-learning',
    version='0.1',
    packages=['libml', 'libml.utils', 'libml.utils.metrics', 'libml.layers', 'libml.models', 'libml.ensemble',
              'libml.ensemble.combiner'],
    url='https://github.com/pdoren/correntropy-and-ensembles-in-deep-learning',
    license='MIT License',
    author='pdoren',
    author_email='pablo.saavedra@ug.uchile.cl',
    description='Library for working with Ensemble Models.'
)
