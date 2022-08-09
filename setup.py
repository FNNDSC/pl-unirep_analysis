from os import path
from setuptools import setup


setup(
    name             = 'unirep_analysis',
    version          = '0.3.0',
    description      = 'An app to run UniRep analysis',
    author           = 'Sandip Samal',
    author_email     = 'dev@babyMRI.org',
    url              = 'http://wiki',
    py_modules       = ['unirep_analysis'],
    packages         = ['src'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.5',
    entry_points     = {
        'console_scripts': [
            'unirep_analysis = unirep_analysis:main'
            ]
        }
)
