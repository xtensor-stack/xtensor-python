import mymodule
import numpy as np

a = np.array([1, 2, 3])
assert np.isclose(np.sum(np.sin(a)), mymodule.sum_of_sines(a))

