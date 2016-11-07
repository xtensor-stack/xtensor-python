import os
import sys
import subprocess

# Build the test extension

here = os.path.abspath(os.path.dirname(__file__))
subprocess.check_call([sys.executable, os.path.join(here, 'setup.py'), 'build_ext', '--inplace'], cwd=here)

# Test it!

from unittest import TestCase
import xtensor_python_test as xt

class ExampleTest(TestCase):

    def test_example(self):
        self.assertEqual(4, xt.test0([4, 5, 6]))
