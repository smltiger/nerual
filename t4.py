import numpy as np


np.random.seed(11)
a = np.random.randn(10)
print(a)

np.random.seed(12)
a = np.random.randn(10)
print(a)

np.random.seed(11)
a = np.random.randn(10)
print(a)
