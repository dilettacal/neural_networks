import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/
"""
def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)\
            .reshape(-1, 28, 28)\
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)


X_train = open_images("../mnist/train-images-idx3-ubyte.gz").reshape(-1, 784)
y_train = open_labels("../mnist/train-labels-idx1-ubyte.gz")
y_train = (y_train == 4).astype(np.float32)


X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")
y_test = (y_test == 4).astype(np.float32)
"""

def S(x):
    return expit(x)
    # return 1 / (1 + np.exp(-x))


def f(w, b, x):
    return S(w @ x.T + b)


def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)))


def J_ableitung_w(w, b, x, y):
    e = f(w, b, x) - y
    #np.sum(x.T * e, axis=1)/x.shape[0]
    return (x.T @ e.T / x.shape[0]).T
#              Fehler im Bezug auf jeden Punkt
#            [[-0.11920292]
#            [ 0.73105858]
#            [-0.26894142]]
#  X-Werte
# [[1 0 0]  dw0 = -0.11920292 * 1 + 0.73105858 * 0 + -0.26894142 * 0
#  [1 1 1]] dw1 = -0.11920292 * 1 + 0.73105858 * 1 + -0.26894142 * 1
#

"""Die Elemente werden spaltenweise zugeordnet. 
w0 = 1 wird der Spalte [1,0,0] und 
w1 = 0 der Spalte [1,1,1] zugeordnet
"""

#Das geht gut für einen Ausgang
#Man hat in neuronalen Netzen aber mehrere Ausgänge
w = np.array([
    [1, 0]
])

b = 1

x = np.array([
    [1, 1],
    [0, 1],
    [0, 1]
])

y = np.array([1, 0, 1])
# [-0.03973431  0.11430475]

#Steigung wird fuer jedes Gewicht berechnet
print(J_ableitung_w(w, b, x, y))

def J_ableitung_b(w, b, x, y):
    return np.mean(f(w, b, x) - y)

