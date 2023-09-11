# -*- coding: utf-8 -*-

""" Setup v6_healthai_survival_analysis_py as a Python package
"""
from os import path
from codecs import open
from setuptools import setup, find_packages


# Get current directory
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Setup the package
setup(
    name='v6_healthai_survival_analysis_py',
    version='1.0.0',
    description='Vantage6 algorithm for 2-years survival for HealthAI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MaastrichtU-CDS/v6-healthai-survival-analysis-py',
    license='MIT License',
    keywords=['Vantage6', 'Federated Algorithms', 'Supervised Learning'],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'vantage6-client',
        'pandas>=1.2.1',
        'scikit-learn>=1.0.2'
    ]
)
