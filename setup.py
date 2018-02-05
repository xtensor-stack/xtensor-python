from setuptools import setup
from glob import glob
import os

# Read version information in include/xtensor-python/xtensor_python_config.hp'
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'include', 'xtensor-python', 'xtensor_python_config.hpp')) as f:
    versions = [line for line in f.readlines() if line.startswith('#define XTENSOR_PYTHON_VERSION_')]
version = versions[0][37:-1] + '.' + versions[1][37:-1] + '.' + versions[2][37:-1]

setup(
    name='xtensor-python',
    version=version,
    long_description='',
    zip_safe=False,
    data_files=[
        ('include/xtensor-python', glob('include/xtensor-python/*.hpp')),
    ],
    install_requires=[
        'xtensor>=0.10.2,==0.10.*',
        'pybind11>=2.1.0,==2.1.*'
    ]
)
