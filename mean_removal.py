#We will use the same input as the binarization
import numpy as np
from sklearn import preprocessing

#Lets define some sample data
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

#Print the mean and standard deviation
print("\nBefore")
print("Mean = ", input_data.mean(axis=0))
print("Standard deviation = ", input_data.std(axis=0))

#Remove mean
data_scaled = preprocessing.scale(input_data)
print("\nAfter")
print("Mean = ", data_scaled.mean(axis=0))
print("Standard deviation = ", data_scaled.std(axis=0))             
