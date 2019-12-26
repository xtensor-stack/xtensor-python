.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Getting started with xtensor-python-cookiecutter
================================================

`xtensor-python-cookiecutter`_ helps extension authors create Python extension modules making use of xtensor.

It takes care of the initial work of generating a project skeleton with

- A complete ``setup.py`` compiling the extension module
- A few examples included in the resulting project including

    - A universal function defined from C++
    - A function making use of an algorithm from the STL on a numpy array
    - Unit tests
    - The generation of the HTML documentation with sphinx

Usage
-----

Install cookiecutter_

.. code::

    pip install cookiecutter

After installing cookiecutter, use the `xtensor-python-cookiecutter`_:

.. code::

    cookiecutter https://github.com/xtensor-stack/xtensor-python-cookiecutter.git

As xtensor-python-cookiecutter runs, you will be asked for basic information about
your custom extension project. You will be prompted for the following
information:

- ``author_name``: your name or the name of your organization,
- ``author_email`` : your project's contact email,
- ``github_project_name``: name of the GitHub repository for your project,
- ``github_organization_name``: name of the GithHub organization for your project,
- ``python_package_name``: name of the Python package created by your extension,
- ``cpp_namespace``: name for the cpp namespace holding the implementation of your extension,
- ``project_short_description``: a short description for your project.
  
This will produce a directory containing all the required content for a minimal extension
project making use of xtensor with all the required boilerplate for package management,
together with a few basic examples.

.. _xtensor-python-cookiecutter: https://github.com/xtensor-stack/xtensor-python-cookiecutter

.. _cookiecutter: https://github.com/audreyr/cookiecutter
