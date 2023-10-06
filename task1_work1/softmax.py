import numpy as np
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

x = np.array([2.0, 1.0, 0.1])
print(x.shape)
print(softmax(x))