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
y_train_encoded = oh.fit_transform(y_train.reshape(-1,1)).toarray()#In Code muss das evtl. transponiert werden


#Test
X_test = open_images(directory_path, test_set, file_type='test_set', level="../").reshape(-1,784)
y_test= open_images(directory_path, test_labels, file_type='test_labels', level="../")


class NeuralNetwork(object):
    def __init__(self, lr = 0.01):
        #Keine pretrained Modelle
        #100 Knoten, 784 Verbindungen (pixel im Bild)
        self.w0 = np.random.randn(100, 784)
        self.w1 = np.random.randn(10,100)
        self.lr = lr

    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)
        e1 = y.T - pred #final error (10,60000)
        e0 = e1.T @ self.w1 #Error auf der Ebene davor

        #UPDATE Regel
        dw1 = e1 * pred *(1-pred) @ a0.T / len(X)#(10,60000) * (60000, 100) = (10, 100)
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X)

        #Update
        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape
        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr *dw0
        print("Kosten: " , self.cost(pred, y))


    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        #Finale Vorhersage
        #Eingang = a0, gewichte = w1
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        """
        #Shapes
        #pred: (10, 10000)
        #y: (10000, 10)
        #Formel:
        #Sum(y-pred)^2 #Least square error
        #Eine der beiden Matrizen muss transponiert werden
        """
        s = (y.T - pred)**2 #(10, 10000)
        #Summe ueber die Zeilen (axis 0)

        #summation = np.sum(s, axis=0)#10000 Ergebnisse
        global_cost = np.mean(np.sum(s, axis=0))
        return global_cost


model = NeuralNetwork()
for i in range(0,100):
    model.train(X_train/255., y_train_encoded)
    # Training für Werte zwischen 0 und 1 --> Teilen durch 255
    y_test_pred = model.predict(X_test / 255.)  # (10,10000), 10000 Zahlen mit jeweils 10 Spalten (?)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    print(np.mean(y_test_pred == y_test))

