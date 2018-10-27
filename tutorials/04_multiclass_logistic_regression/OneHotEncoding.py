import numpy as np

y_train = np.array([1,5,7,5,6,2,5,7,3,5,6,9])

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

try:
    ohe.fit(y_train)  # Daten "kennen"
    ohe.transform(y_train) #Transformation !
except ValueError as ve:
    print(ve)
#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
print(y_train)
y_train = np.array([1,5,7,5,6,2,5,7,3,5,6,9,0]).reshape(-1,1) #Matrix mit einer Spalte und jede Zeile entspricht einem Datensatz
print(y_train)
ohe.fit(y_train)  # Daten "kennen"
encoded = ohe.transform(y_train) #Transformation !
print(encoded)
"""Es wird eine sparse matrix erzeugt. Eine Spalte ist immer auf 1 gesetzt, je nach Zahl"""
encoded_as_array = encoded.toarray()
print(encoded_as_array)
