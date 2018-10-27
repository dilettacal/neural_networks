from MNIST_explore import open_images
import LogisticRegressionMNIST as lrMNIST
import numpy as np
import matplotlib.pyplot as plt

directory_path = "data/mnist"
train_set = "train-images-idx3-ubyte.gz" #60000 Bilder 28*28
train_labels = "train-labels-idx1-ubyte.gz"

test_set = "t10k-images-idx3-ubyte.gz"
test_labels = "t10k-labels-idx1-ubyte.gz"

#Open dataset
X_train_set = open_images(directory_path, train_set, file_type='train_set', level="../")
y_train_set = open_images(directory_path, train_labels, file_type='train_labels', level="../")

X_train_set = open_images(directory_path, test_set, file_type='train_set', level="../")
y_train_set = open_images(directory_path, test_labels, file_type='train_labels', level="../")

#Test
X_test_set = open_images(directory_path, train_set, file_type='train_set', level="../")
y_test_set = open_images(directory_path, train_labels, file_type='train_labels', level="../")

print("Ein Bild aus Datensatz: ")
print(X_train_set[0]) # Ein Datum besteht aus 28*28 Achsen
""" Ein nicht verarbeitetes Datum ist ein gesamtes 3D Array.
    Ziel: Eine Matrix erzeugen, wo jedes Bild einer Zeile entspricht """

print("Reshaping X Datensatz: ")
img_size = 28*28 #784
X_train = X_train_set.reshape(-1,img_size) #60000 Zeilen, jeweils mit 784 Spalten
X_test = X_test_set.reshape(-1,img_size)
print(X_train.shape)
print("Erstes reshaped Bild: ")
print(X_train[0]) #Eine Zeile enthaelt ein ganzes Bild
print("*"*25)
""" Die Logistische Regression braucht einen Labeldatensatz mit boolschen Werten
    und funktioniert lediglich fuer nur eine einzelne Klasse.
    Zum Beispiel man trainiert das Modell, damit es Prediction auf eine Zahl (bzw. eine Klasse) machen kann.
    Training auf die Klasse 4 --> Reshape y-Datensatz mit float-Werten bezueglich der Zahl 4
    """
print("Reshaping Y Datensatz: ")
klasse = 4

y_train = (y_train_set == klasse).astype(np.float32)
y_test = (y_test_set == klasse).astype(np.float32)
print(y_train[:10])

#Parameters
lr = 0.00001
b = 1 #Bias
#Weights
#Wenn man eine Null hat, dann ist die Steigung am groessten
w = np.zeros((1, 784))
""" Train model """
(costs, weights, bias) = lrMNIST.train(w,b,X_train,y_train,lr, iterations=50)

""" Prediction on model """
(y_test_pred, mean_pred) = lrMNIST.predict(w,b, X_test, y_test)

history = lrMNIST.accuracy_history(w,b, X_test, y_test)
print("History: ")
print(history)
print("Final accuracy: " + str(mean_pred))

plt.plot(costs)
plt.show()


