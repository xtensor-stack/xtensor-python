from benchmark_xtensor_python import sum_array
import numpy as np

u = np.ones(1000000, dtype=float)
from timeit import timeit
print (timeit ('sum_array(u)', setup='from __main__ import u, sum_array', number=1000))
