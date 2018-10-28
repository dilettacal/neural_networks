import gzip
import numpy as np
import matplotlib.pyplot as plt
import os

def open_images(directory_path, filename, file_type='', level="./"):
    current_directory = os.getcwd()
    base_name = os.path.basename(current_directory)
    dir_name = os.path.dirname(current_directory)
    path = level + directory_path
    os.chdir(path)
    pixel = 28
    #Read file as binary file
    with gzip.open(filename, 'rb') as file:
        data = file.read()
        if file_type == 'train_set' or file_type == "test_set":
            #Create a numpy array
            #Format uint8
            #From buffer == eindimensionales Array
            #Reshape --> 3D Array (Bildformat)
            read_data = np.frombuffer(data, dtype=np.uint8, offset=16)\
                .reshape(-1,pixel,pixel)\
                .astype(np.float32) #Umwandlung nach float
        elif file_type == "train_labels" or file_type == "test_labels":
            read_data = np.frombuffer(data, dtype=np.uint8, offset=8)
        else:
            raise Exception('File type not valid!')

    if level == "./":
        os.chdir(current_directory)
    elif level == "../":
        back_path = level*2+base_name
        os.chdir(back_path)
    return read_data


def run():
    print("Read x und y (train set)")
    X_train = open_images(directory_path, train_set, file_type='train_set', level="../")
    y_train = open_images(directory_path, train_labels, file_type='train_labels', level="../")

    print("Datenerkundung: ")
    print("Datensatz shape: ")
    print(X_train.shape)
    print("Shape von einem Bild im Datensatz: ")
    print(X_train[1].shape)
    print("Erstes Bild: ")
    print(X_train[0])
    print("Visualisiere Bild 1:")
    # Drinnen gibt es Helligkeitswerte
    plt.imshow(X_train[1])
    plt.show()

    print("*" * 50)
    print("Label shape: ")
    print(y_train.shape)
    print(y_train[1])

#paths
directory_path = "data/mnist"
train_set = "train-images-idx3-ubyte.gz" #60000 Bilder 28*28
train_labels = "train-labels-idx1-ubyte.gz"

X_train = open_images(directory_path, train_set, file_type='train_set', level="../").reshape(-1,784)
y_train = open_images(directory_path, train_labels, file_type='train_labels', level="../")

print(X_train[0].shape) #(784,) 784 Zeilen hintereinander
print(X_train[0].reshape(28,28).shape) #(28,28)

#image = X_train[0].reshape(28,28)
#plt.imshow(np.roll(image,(4,5), axis=(0,1)))
#plt.show()

#Anwendung auf gesamtem Datensatz
#(-1,28,28)
#Dim0: Wie viele Bilder == 60000
#Jedes Bild hat dim1 und dim2, wie viele Pixel für jedes Bild in der jeweiligen Dimension
images = X_train.reshape(-1,28,28)
images = np.roll(images, (3,3), axis=(1,2)) #Roll bezieht sich auf die Pixel
plt.imshow(images[1])
plt.show()