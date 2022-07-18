from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.rst')) as f:
    readme = f.read()

setup(
    name             = 'unirep_analysis',
    version          = '0.1.1',
    description      = 'An app to ...',
    long_description = readme,
    author           = 'Sandip Samal',
    author_email     = 'dev@babyMRI.org',
    url              = 'http://wiki',
    packages         = ['unirep_analysis','analysis','data'],
    install_requires = ['chrisapp'],
    test_suite       = 'nose.collector',
    tests_require    = ['nose'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.5',
    entry_points     = {
        'console_scripts': [
            'unirep_analysis = unirep_analysis.__main__:main'
            ]
        }
)
