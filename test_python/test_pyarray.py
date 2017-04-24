import os
import sys
import subprocess

# Build the test extension

here = os.path.abspath(os.path.dirname(__file__))
subprocess.check_call([sys.executable, os.path.join(here, 'setup.py'), 'build_ext', '--inplace'], cwd=here)

# Test it!

from unittest import TestCase
import xtensor_python_test as xt
import numpy as np

class ExampleTest(TestCase):

    def test_example1(self):
        self.assertEqual(4, xt.example1([4, 5, 6]))

    def test_example2(self):
        x = np.array([[0., 1.], [2., 3.]])
        res = np.array([[2., 3.], [4., 5.]])
        y = xt.example2(x)
        np.testing.assert_allclose(y, res, 1e-12)

    def test_vectorize(self):
        x1 = np.array([[0, 1], [2, 3]])
        x2 = np.array([0, 1])
        res = np.array([[0, 2], [2, 4]])
        y = xt.vectorize_example1(x1, x2)
        np.testing.assert_array_equal(y, res)

    def test_readme_example1(self):
        v = np.arange(15).reshape(3, 5)
        y = xt.readme_example1(v)
        np.testing.assert_allclose(y, 1.2853996391883833, 1e-12)

    def test_complex_overload_reg(self):
        a = 23.23
        c = 2.0 + 3.1j
        self.assertEqual(xt.complex_overload_reg(a), a)
        self.assertEqual(xt.complex_overload_reg(c), c)

    def test_complex_overload(self):
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 3)
        c = a + b * 1j
        y = xt.complex_overload(c)
        print(y)
        np.testing.assert_allclose(np.imag(y), np.imag(c))
        np.testing.assert_allclose(np.real(y), np.real(c))
        x = xt.complex_overload(b)
        self.assertEqual(x.dtype, b.dtype)
        np.testing.assert_allclose(x, b)

    def test_readme_example2(self):
        x = np.arange(15).reshape(3, 5)
        y = [1, 2, 3, 4, 5]
        z = xt.readme_example2(x, y)
        np.testing.assert_allclose(z, 
            [[-0.540302,  1.257618,  1.89929 ,  0.794764, -1.040465],
             [-1.499227,  0.136731,  1.646979,  1.643002,  0.128456],
             [-1.084323, -0.583843,  0.45342 ,  1.073811,  0.706945]], 1e-5)

    def test_rect_to_polar(self):
        x = np.ones(10, dtype=complex)
        z = xt.rect_to_polar(x[::2]);
        np.testing.assert_allclose(z, np.ones(5, dtype=float), 1e-5)

    def test_shape_comparison(self):
        x = np.ones([4, 4])
        y = np.ones([5, 5])
        z = np.zeros([4, 4])
        self.assertFalse(xt.compare_shapes(x, y))
        self.assertTrue(xt.compare_shapes(x, z))

