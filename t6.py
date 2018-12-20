import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=160)
np.set_printoptions(formatter={'float':'{:0.3f}'.format})

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

a = np.array([125.297, -125.713, -1339.269, -127.178, -1307.343 ,-124.929, -903.338, -125.063, -126.001, -125.387, -125.086, -125.917])

b = a - (-263)
print(a)
print(b)
c = abs(b.mean())

print(b/(c*12))
