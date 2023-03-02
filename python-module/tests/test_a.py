import unittest
import timeit
import warnings

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

        n = timeit.timeit(lambda: np.mean(a), number=10)
        x = timeit.timeit(lambda: xt.mean(a), number=10)

        if x / n > 1.1:
            warnings.warn(f"efficiency xt.mean {x / n:.2e}")

    def test_flip(self):

        axis = int(np.random.randint(0, high=3))
        a = np.random.random([103, 102, 101])
        n = np.flip(a, axis)
        x = xt.flip(a, axis)

        self.assertTrue(np.allclose(n, x))

        n = timeit.timeit(lambda: np.flip(a, axis), number=10)
        x = timeit.timeit(lambda: xt.flip(a, axis), number=10)

        if x / n > 1.1:
            warnings.warn(f"efficiency xt.flip {x / n:.2e}")

    def test_cos(self):

        a = np.random.random([103, 102, 101])
        n = np.cos(a)
        x = xt.cos(a)

        self.assertTrue(np.allclose(n, x))

        n = timeit.timeit(lambda: np.cos(a), number=10)
        x = timeit.timeit(lambda: xt.cos(a), number=10)

        if x / n > 1.1:
            warnings.warn(f"efficiency xt.cos {x / n:.2e}")



if __name__ == "__main__":

    unittest.main()
