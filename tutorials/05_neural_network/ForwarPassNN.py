from MNIST_explore import open_images
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
import pickle
import numpy as np


directory_path = "data/mnist"
train_set = "train-images-idx3-ubyte.gz" #60000 Bilder 28*28
train_labels = "train-labels-idx1-ubyte.gz"

test_set = "t10k-images-idx3-ubyte.gz"
test_labels = "t10k-labels-idx1-ubyte.gz"

#Open dataset
X_train = open_images(directory_path, train_set, file_type='train_set', level="../").reshape(-1,784)
y_train = open_images(directory_path, train_labels, file_type='train_labels', level="../")

oh = OneHotEncoder()
y_train = oh.fit_transform(y_train.reshape(-1,1)).toarray().T #Transponiert, somit kann das problemlos subtrahiert werden


#Test
X_test = open_images(directory_path, test_set, file_type='test_set', level="../").reshape(-1,784)
y_test= open_images(directory_path, test_labels, file_type='test_labels', level="../")


class NeuralNetwork(object):
    def __init__(self):
        #Load model
        with open("./weights/w0.p", "rb") as file:
            self.w0 = pickle.load(file)
        with open("./weights/w1.p", "rb") as file:
            self.w1 = pickle.load(file)

    def activation(self, x):
        return expit(x)

    def predict(self, X):
        #VOrhersage durch Logistische Reg
        #Umgekehrt geht auch
        a0 = self.activation(self.w0 @ X.T)
        #Finale Vorhersage
        #Eingang = a0, gewichte = w1
        pred = self.activation(self.w1 @ a0)
        return pred


model = NeuralNetwork()
print(model.w0.shape) #(100,784) = 784 pixel = 784 Eingänge. 100 Ausgänge
print(model.w1.shape) #(10, 100) = 100 Eingänge (Ausgänge von w0) und 10 Ausgänge

#Training für Werte zwischen 0 und 1 --> Teilen durch 255
y_test_pred = model.predict(X_test/255.)#(10,10000), 10000 Zahlen mit jeweils 10 Spalten (?)
y_test_pred = np.argmax(y_test_pred, axis=0)
print(np.mean(y_test_pred == y_test))