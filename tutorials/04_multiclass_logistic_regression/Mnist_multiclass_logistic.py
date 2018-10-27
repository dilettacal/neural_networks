from MNIST_explore import open_images
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import MultiClassLogisticRegression as mclr

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

print(X_test.shape)
print(y_test.shape)

lr = 0.00001
w = np.zeros((10,784))
b = np.ones(10)

(costs, weights, bias) = mclr.train(w,b,X_train,y_train,lr, iterations=150)

preds, accs, ff = mclr.predict(weights,bias, X_test, y_test)
print("Model accuracy: ", ff)#ca 86% mit 100 Durchläufen

plt.plot(accs)
plt.ylabel("Accuracy %")
plt.xlabel("Iterations")
plt.show()