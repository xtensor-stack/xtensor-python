############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

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

class XtensorTest(TestCase):
    def test_rm(self):
        xt.test_rm(np.array([10], dtype=int))

    def test_example1(self):
        self.assertEqual(4, xt.example1([4, 5, 6]))

    def test_example2(self):
        x = np.array([[0., 1.], [2., 3.]])
        res = np.array([[2., 3.], [4., 5.]])
        y = xt.example2(x)
        np.testing.assert_allclose(y, res, 1e-12)

    def test_example3(self):
        x = np.arange(2 * 3).reshape(2, 3)
        xc = np.asfortranarray(x)
        y = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        v = y[1:, 1:, 0]
        z = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        np.testing.assert_array_equal(xt.example3_xarray(x), x.T + 2)
        np.testing.assert_array_equal(xt.example3_xarray_colmajor(xc), xc.T + 2)
        np.testing.assert_array_equal(xt.example3_xtensor3(y), y.T + 2)
        np.testing.assert_array_equal(xt.example3_xtensor2(x), x.T + 2)
        np.testing.assert_array_equal(xt.example3_xtensor2(y[1:, 1:, 0]), v.T + 2)
        np.testing.assert_array_equal(xt.example3_xtensor2_colmajor(xc), xc.T + 2)

        np.testing.assert_array_equal(xt.example3_xfixed3(y), y.T + 2)
        np.testing.assert_array_equal(xt.example3_xfixed2(x), x.T + 2)
        np.testing.assert_array_equal(xt.example3_xfixed2_colmajor(xc), xc.T + 2)

        with self.assertRaises(TypeError):
            xt.example3_xtensor3(x)

        with self.assertRaises(TypeError):
            xt.example3_xfixed3(x)

        with self.assertRaises(TypeError):
            x = np.arange(3*2).reshape(3, 2)
            xt.example3_xfixed2(x)
    def test_broadcast_addition(self):
        x = np.array([[2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4.],
                      [1., 2., 3., 4.],
                      [1., 2., 3., 4.]])
        res = np.array([[3., 5., 7., 9.],
                        [3., 5., 7., 9.],
                        [3., 5., 7., 9.]])
        z = xt.array_addition(x, y)
        np.testing.assert_allclose(z, res, 1e-12)
    def test_broadcast_subtraction(self):
        x = np.array([[4., 5., 6., 7.]])
        y = np.array([[4., 3., 2., 1.],
                      [4., 3., 2., 1.],
                      [4., 3., 2., 1.]])
        res = np.array([[0., 2., 4., 6.],
                        [0., 2., 4., 6.],
                        [0., 2., 4., 6.]])
        z = xt.array_subtraction(x, y)
        np.testing.assert_allclose(z, res, 1e-12)

    def test_broadcast_multiplication(self):
        x = np.array([[1., 2., 3., 4.]])
        y = np.array([[3., 2., 3., 2.],
                      [3., 2., 3., 2.],
                      [3., 2., 3., 2.]])
        res = np.array([[3., 4., 9., 8.],
                        [3., 4., 9., 8.],
                        [3., 4., 9., 8.]])
        z = xt.array_multiplication(x, y)
        np.testing.assert_allclose(z, res, 1e-12)

    def test_broadcast_division(self):
        x = np.array([[8., 6., 4., 2.]])
        y = np.array([[2., 2., 2., 2.],
                      [2., 2., 2., 2.],
                      [2., 2., 2., 2.]])
        res = np.array([[4., 3., 2., 1.],
                        [4., 3., 2., 1.],
                        [4., 3., 2., 1.]])
        z = xt.array_division(x, y)
        np.testing.assert_allclose(z, res, 1e-12)

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

    def test_int_overload(self):
        for dtype in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]:
            b = xt.int_overload(np.ones((10), dtype))
            self.assertEqual(str(dtype.__name__), b)

    def test_dtype(self):
        var = xt.dtype_to_python()
        self.assertEqual(var.dtype.names, ('a', 'b', 'c', 'x'))

        exp_dtype = {
             'a': (np.dtype('float64'), 0),
             'b': (np.dtype('int32'), 8),
             'c': (np.dtype('int8'), 12),
             'x': (np.dtype(('<f8', (3,))), 16)
        }

        self.assertEqual(var.dtype.fields, exp_dtype)

        self.assertEqual(var[0]['a'], 123)
        self.assertEqual(var[0]['b'], 321)
        self.assertEqual(var[0]['c'], ord('a'))
        self.assertTrue(np.all(var[0]['x'] == [1, 2, 3]))

        self.assertEqual(var[1]['a'], 111)
        self.assertEqual(var[1]['b'], 222)
        self.assertEqual(var[1]['c'], ord('x'))
        self.assertTrue(np.all(var[1]['x'] == [5, 5, 5]))

        d_dtype = np.dtype({'names':['a','b'], 'formats':['<f8','<i4'], 'offsets':[0,8], 'itemsize':16})

        darr = np.array([(1, ord('p')), (123, ord('c'))], dtype=d_dtype)
        self.assertEqual(darr[0]['a'], 1)
        res = xt.dtype_from_python(darr)
        self.assertEqual(res[0]['a'], 123.)
        self.assertEqual(darr[0]['a'], 123.)

    def test_char_array(self):
        var = np.array(['hello', 'from', 'python'], dtype=np.dtype('|S20'));
        self.assertEqual(var[0], b'hello')
        xt.char_array(var)
        self.assertEqual(var[0], b'hello')
        self.assertEqual(var[1], b'from')
        self.assertEqual(var[2], b'c++')

    def test_col_row_major(self):
        var = np.arange(50, dtype=float).reshape(2, 5, 5)

        with self.assertRaises(RuntimeError):
            xt.col_major_array(var)

        with self.assertRaises(TypeError):
            xt.row_major_tensor(var.T)

        with self.assertRaises(TypeError):
            xt.row_major_tensor(var[:, ::2, ::2])

        with self.assertRaises(TypeError):
            # raise for wrong dimension
            xt.row_major_tensor(var[0, 0, :])

        xt.row_major_tensor(var)
        varF = np.arange(50, dtype=float).reshape(2, 5, 5, order='F')
        xt.col_major_array(varF)
        xt.col_major_array(varF[:, :, 0]) # still col major!

    def test_xscalar(self):
        var = np.arange(50, dtype=int)
        self.assertTrue(np.sum(var) == xt.xscalar(var))

    def test_bad_argument_call(self):
        with self.assertRaises(TypeError):
            xt.simple_array("foo")

        with self.assertRaises(TypeError):
            xt.simple_tensor("foo")

    def test_diff_shape_overload(self):
        self.assertEqual(1, xt.diff_shape_overload(np.ones(2)))
        self.assertEqual(2, xt.diff_shape_overload(np.ones((2, 2))))

        with self.assertRaises(TypeError):
            # FIXME: the TypeError information is not informative
            xt.diff_shape_overload(np.ones((2, 2, 2)))

    def test_native_casters(self):
        import gc

        # check keep alive policy for get_strided_view()
        gc.collect()
        obj = xt.test_native_casters()
        a = obj.get_strided_view()
        obj = None
        gc.collect()
        _ = np.zeros((100, 100))
        self.assertEqual(a.sum(), a.size)

        # check keep alive policy for get_array_adapter()
        gc.collect()
        obj = xt.test_native_casters()
        a = obj.get_array_adapter()
        obj = None
        gc.collect()
        _ = np.zeros((100, 100))
        self.assertEqual(a.sum(), a.size)

        # check keep alive policy for get_array_adapter()
        gc.collect()
        obj = xt.test_native_casters()
        a = obj.get_tensor_adapter()
        obj = None
        gc.collect()
        _ = np.zeros((100, 100))
        self.assertEqual(a.sum(), a.size)

        # check keep alive policy for get_owning_array_adapter()
        gc.collect()
        obj = xt.test_native_casters()
        a = obj.get_owning_array_adapter()
        gc.collect()
        _ = np.zeros((100, 100))
        self.assertEqual(a.sum(), a.size)

        # check keep alive policy for view_keep_alive_member_function()
        gc.collect()
        a = np.ones((100, 100))
        b = obj.view_keep_alive_member_function(a)
        obj = None
        a = None
        gc.collect()
        _ = np.zeros((100, 100))
        self.assertEqual(b.sum(), b.size)

        # check shared buffer (insure that no copy is done)
        obj = xt.test_native_casters()
        arr = obj.get_array()

        strided_view = obj.get_strided_view()
        strided_view[0, 1] = -1
        self.assertEqual(strided_view.shape, (1, 2))
        self.assertEqual(arr[0, 2], -1)

        adapter = obj.get_array_adapter()
        self.assertEqual(adapter.shape, (2, 2))
        adapter[1, 1] = -2
        self.assertEqual(arr[0, 5], -2)

        adapter = obj.get_tensor_adapter()
        self.assertEqual(adapter.shape, (2, 2))
        adapter[1, 1] = -3
        self.assertEqual(arr[0, 5], -3)

class AttributeTest(TestCase):

    def setUp(self):
        self.c = xt.C()

    def test_copy(self):
        arr = self.c.copy
        arr[0] = 1
        self.assertEqual([0.]*4, self.c.copy.tolist())

    def test_reference(self):
        arr = self.c.ref
        arr[0] = 1
        self.assertEqual([1.] + [0.]*3, self.c.ref.tolist())
