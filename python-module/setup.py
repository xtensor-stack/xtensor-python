from pathlib import Path

from setuptools_scm import get_version
from skbuild import setup

project_name = "xt"

this_directory = Path(__file__).parent
long_description = (this_directory / ".." / "README.md").read_text()
license = (this_directory / ".." / "LICENSE").read_text()

setup(
    name=project_name,
    description="xt - Python bindings of xtensor",
    long_description=long_description,
    version=get_version(root="..", relative_to=__file__),
    license=license,
    author="Tom de Geus",
    url=f"ttps://github.com/xtensor-stack/{project_name}",
    packages=[f"{project_name}"],
    package_dir={"": "module"},
    cmake_install_dir=f"module/{project_name}",
    cmake_minimum_required_version="3.13",
)
