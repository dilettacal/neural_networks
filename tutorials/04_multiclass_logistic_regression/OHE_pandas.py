import numpy as np
import pandas as pd

y_train = np.array(["Berlin", "Muenchen", "Koeln", "Hamburg"])
print(y_train)

#Vearbeitung mit pandas
data = pd.get_dummies(y_train)
print(data)

values = data.values #Umwandlung nach numpy array
print(values)

""" Anwendung auf csv Dateien """
df = pd.read_csv("autos_prepared.csv", encoding='utf8')
df_encoded = pd.get_dummies(df, columns=['fuelType'])

print(df_encoded)