import mymodule
import numpy as np

c = np.array([[1, 2, 3], [4, 5, 6]])
assert np.isclose(np.sum(np.sin(c)), mymodule.sum_of_sines(c))
assert np.isclose(np.sum(np.cos(c)), mymodule.sum_of_cosines(c))
