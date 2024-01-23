import unittest

import xt
import numpy as np


class test_a(unittest.TestCase):
    """
    ??
    """

    def test_mean(self):

        a = np.random.random([103, 102, 101])
        n = np.mean(a)
        x = xt.mean(a)

        self.assertTrue(np.allclose(n, x))

    def test_average(self):

        a = np.random.random([103, 102, 101])
        w = np.random.random([103, 102, 101])
        n = np.average(a, weights=w)
        x = xt.average(a, w)

        self.assertTrue(np.allclose(n, x))

    def test_average_axes(self):

        a = np.random.random([103, 102, 101])
        w = np.random.random([103, 102, 101])
        axis = int(np.random.randint(0, high=3))
        n = np.average(a, weights=w, axis=(axis,))
        x = xt.average(a, w, [axis])

        self.assertTrue(np.allclose(n, x))

    def test_flip(self):

        axis = int(np.random.randint(0, high=3))
        a = np.random.random([103, 102, 101])
        n = np.flip(a, axis)
        x = xt.flip(a, axis)

        self.assertTrue(np.allclose(n, x))

    def test_cos(self):

        a = np.random.random([103, 102, 101])
        n = np.cos(a)
        x = xt.cos(a)

        self.assertTrue(np.allclose(n, x))


if __name__ == "__main__":

    unittest.main()
