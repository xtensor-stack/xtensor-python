import unittest
import timeit
import warnings
from prettytable import PrettyTable

import xt
import numpy as np


class test_a(unittest.TestCase):
    """
    ??
    """

    @classmethod
    def setUpClass(self):

        self.a = np.random.random([103, 102, 101])
        self.axis = int(np.random.randint(0, high=3))
        self.data = []

    @classmethod
    def tearDownClass(self):

        self.data = sorted(self.data, key=lambda x: x[1])

        table = PrettyTable(["function", "xtensor / numpy"])

        for row in self.data:
            table.add_row(row)

        print("")
        print(table)
        print("")

        # code = table.get_html_string()
        # with open('profile.html', 'w') as file:
        #     file.write(code)

    def test_mean(self):

        n = timeit.timeit(lambda: np.mean(self.a), number=10)
        x = timeit.timeit(lambda: xt.mean(self.a), number=10)
        self.data.append(("mean", x / n))

    def test_flip(self):

        n = timeit.timeit(lambda: np.flip(self.a, self.axis), number=10)
        x = timeit.timeit(lambda: xt.flip(self.a, self.axis), number=10)
        self.data.append(("flip", x / n))

    def test_cos(self):

        n = timeit.timeit(lambda: np.cos(self.a), number=10)
        x = timeit.timeit(lambda: xt.cos(self.a), number=10)
        self.data.append(("cos", x / n))

    def test_isin(self):

        a = (np.random.random([103, 102]) * 1000).astype(int)
        b = (np.random.random([103, 102]) * 1000).astype(int)

        n = timeit.timeit(lambda: np.isin(a, b), number=10)
        x = timeit.timeit(lambda: xt.isin(a, b), number=10)
        self.data.append(("isin", x / n))

    def test_in1d(self):

        a = (np.random.random([1003]) * 1000).astype(int)
        b = (np.random.random([1003]) * 1000).astype(int)

        n = timeit.timeit(lambda: np.in1d(a, b), number=10)
        x = timeit.timeit(lambda: xt.in1d(a, b), number=10)
        self.data.append(("in1d", x / n))


if __name__ == "__main__":

    unittest.main()
