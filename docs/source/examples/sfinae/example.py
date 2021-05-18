import mymodule
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
b = np.array(a, copy=True)
mymodule.times_dimension(b) # changing in-place!
assert np.allclose(2 * a, b)

