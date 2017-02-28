from benchmark_xtensor_python import rect_to_polar
import numpy as np

from timeit import timeit
w = np.ones(100000, dtype=complex)
print (timeit('rect_to_polar(w[::2])', 'from __main__ import w, rect_to_polar', number=1000))
