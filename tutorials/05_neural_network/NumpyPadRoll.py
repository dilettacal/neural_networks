import numpy as np

a = np.array([1,2,3])

#Pad
#(2,2) = Anzahl der Nullen (mode="constant"), rechts und links
print(np.pad(a,((2,2)), mode="constant"))

#Roll
print(np.roll(a, (1,))) #3 1 2
print(np.roll(a, (2,))) #2 3 1