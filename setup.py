#!/usr/bin/env python

# Setup script for PyPI; use CMakeFile.txt to build extension modules

from setuptools import setup
from distutils.command.install_headers import install_headers
from distutils.command.build_py import build_py
from xtensor_python import __version__
import os

package_data = [
    'include/xtensor-python/pyarray.hpp',
    'include/xtensor-python/pycontainer.hpp',
    'include/xtensor-python/pytensor.hpp',
    'include/xtensor-python/xtensor_python_config.hpp',
    'include/xtensor-python/pyarray_backstrides.hpp',
    'include/xtensor-python/pystrides_adaptor.hpp',
    'include/xtensor-python/pyvectorize.hpp',
    'include/xtensor-python/xtensor_type_caster_base.hpp',
]

# Prevent installation of pybind11 headers by setting
# PYBIND11_USE_CMAKE.
if os.environ.get('PYBIND11_USE_CMAKE'):
    headers = []
else:
    headers = package_data


class InstallHeaders(install_headers):
    """Use custom header installer because the default one flattens subdirectories"""
    def run(self):
        if not self.distribution.headers:
            return

        for header in self.distribution.headers:
            subdir = os.path.dirname(os.path.relpath(header, 'include/xtensor-python'))
            install_dir = os.path.join(self.install_dir, subdir)
            self.mkpath(install_dir)

            (out, _) = self.copy_file(header, install_dir)
            self.outfiles.append(out)


# Install the headers inside the package as well
class BuildPy(build_py):
    def build_package_data(self):
        build_py.build_package_data(self)
        for header in package_data:
            target = os.path.join(self.build_lib, 'xtensor_python', header)
            self.mkpath(os.path.dirname(target))
            self.copy_file(header, target, preserve_mode=False)


setup(
    name='xtensor-python',
    version=__version__,
    description='',
    author='',
    author_email='',
    url='https://github.com/xtensor-stack/xtensor-python',
    download_url='https://github.com/pybind/pybind11/tarball/v' + __version__,
    packages=['xtensor_python'],
    license='BSD',
    headers=headers,
    zip_safe=False,
    cmdclass=dict(install_headers=InstallHeaders, build_py=BuildPy),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: BSD License'
    ],
    keywords='C++11, Python bindings',
    long_description=""".""")
