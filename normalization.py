#We will use the same input as the binarization
import numpy as np
from sklearn import preprocessing

#Lets define some sample data
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

#Normalize data
data_mormalized_l1 = preprocessing.normalize(input_data, norm="l1")
data_mormalized_l2 = preprocessing.normalize(input_data, norm="l2")

print("\n Normalize data L1 \n", data_mormalized_l1)
print("\n Normalize data L2 \n", data_mormalized_l2)
