# Requirement

 - pybind11 master (not 1.8.1)
 - xtensor 0.1.0

# Installation

 - build and install conda recipe

# Dev installation

 - symlink `include\xtensor-python` to `$SYS_PREFIX\include\xtensor-python`

# Testing

  Testing `xtensor-python` requires `nosetests`

  ``` bash
  nosetests .
  ```

  To pick up changes in `xtensor-python` while rebuilding, delete the `build/` directory. 
