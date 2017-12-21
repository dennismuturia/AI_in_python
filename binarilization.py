import numpy as np
from sklearn import preprocessing

#Lets define some sample data
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

#Binerized data
data_binerized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\n Binerized data: \n", data_binerized)
