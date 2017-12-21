#We will use the same input as the binarization
import numpy as np
from sklearn import preprocessing

#Lets define some sample data
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

#Min max scaling
data_scalar_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scalar_minmax = data_scalar_minmax.fit_transform(input_data)

print("\n Data scaled min max is:", data_scalar_minmax)
