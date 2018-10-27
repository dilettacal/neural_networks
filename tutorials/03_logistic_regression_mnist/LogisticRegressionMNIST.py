import numpy as np
from scipy.special import expit

""" Logistic regression implementation for MNIST dataset"""

#Sigmoid function 1 / (1 + np.exp(-x))
def S(x):
    return expit(x)

#Hypothesis
def f(w, b, x):
    return S(w @ x.T + b)


#Costfunction
def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)))


#Partial derivatives
#Partial derivative weights
def J_ableitung_w(w, b, x, y):
    e = f(w, b, x) - y
    return np.mean(x.T * e, axis=1)


#Partial derivative bias
def J_ableitung_b(w, b, x, y):
    return np.mean(f(w, b, x) - y)

def train(w,b,X, y, reg, iterations = 200):
    weights = []
    bias = []
    costs = []
    cost = J(w, b, X, y)
    weights.append(w)
    bias.append(b)
    costs.append(cost)

    for i in range(0, iterations):

        dw = J_ableitung_w(w, b, X, y)
        db = J_ableitung_b(w, b, X, y)

        w = w - reg * dw
        weights.append(w)
        b = b - reg * db
        bias.append(b)

        cost = J(w, b, X, y)
        print("Kosten: " + str(cost))
        costs.append(cost)
    return costs, weights, bias


def predict(w,b, X_test, y_test):
    y_test_pred = f(w, b, X_test) > 0.5 #Array  mit True/False
    y_test_pred = y_test_pred.reshape(-1)
    mean_pred = np.mean(y_test == y_test_pred)
    return y_test_pred, mean_pred

def accuracy_history(w,b, X_test, y_test):
    history = []
    print(w[0].shape)
    for weight in w:
        _, mean_pred = predict(weight, b, X_test, y_test)
        print(mean_pred)
        history.append(np.mean(mean_pred))
    return history



