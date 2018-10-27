import numpy as np
from scipy.special import expit
from scipy.misc import imread

""" Multiclass Logistic regression implementation for MNIST dataset"""

#Sigmoid function 1 / (1 + np.exp(-x))
def S(x):
    return expit(x)

#Hypothesis
def f(w, b, x):
    #Problem: b muss die selbe shape haben wie die Matrix !
    #b wird auf jede Matrixzeile angewendet
    #Matrix wird transponiert --> x (10,60000) --> x.T (60000, 10)
    a = w @ x.T
    return S(a.T + b).T


#Costfunction
def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)), axis=1)


#Partial derivatives
#Partial derivative weights
def J_ableitung_w(w, b, x, y):
    e = f(w, b, x) - y
    #Gleichbedeutend wie: (x.T @ e.T / x.shape[0]).T
    #return np.sum(x.T * e, axis=1)/x.shape[0]
    return (x.T @ e.T / x.shape[0]).T


#Partial derivative bias
def J_ableitung_b(w, b, x, y):
    #ANgeben ob mean für Zeilen oder Spalten berechnet werden muss!
    return np.mean(f(w, b, x) - y, axis=1) #shape = (10,)

def train(w,b,X, y, reg, iterations = 200):
    weights = []
    bias = []
    costs = []
    #cost = J(w, b, X, y)
    #weights.append(w)
    #bias.append(b)
    #costs.append(cost)

    for i in range(0, iterations):

        dw = J_ableitung_w(w, b, X, y) #(10, 784), 10 Modelle und 784 Steigungen zu den entsprechenden ws
        db = J_ableitung_b(w, b, X, y) #(10,)
        w = w - reg * dw
        weights.append(w)
        b = b - reg * db
        bias.append(b)

        cost = J(w, b, X, y) #(10,) --> Loss fuer jedes Modell
        print("Kosten: " + str(cost))
        costs.append(cost)
    return costs, weights, bias


def predict(w,b, X_test, y_test):
    y_test_pred = []
    accuracies = []
    assert len(w) == len(b)
    for i in range(len(w)):
        pred = f(w[i], b[i], X_test) #(10,10000) --> 10 Wahrscheinlichkeiten, jeweils fuer ein Modell
        pred = np.argmax(pred, axis=0) #geschaetzte y Daten
        accuracy = np.mean(pred == y_test)
        accuracies.append(accuracy)
        y_test_pred.append(pred)
    final_acc = np.mean(accuracies)
    return y_test_pred, accuracies, final_acc

