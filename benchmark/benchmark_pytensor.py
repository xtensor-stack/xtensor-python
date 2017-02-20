from benchmark_xtensor_python import sum_tensor
import numpy as np

u = np.ones(1000000, dtype=float)
#print(sum_tensor(u))
from timeit import timeit
print (timeit ('sum_tensor(u)', setup='from __main__ import u, sum_tensor', number=1000))
