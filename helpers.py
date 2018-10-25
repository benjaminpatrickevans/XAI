
import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder


def read_data(data_path, class_index="last"):
    # Load the data
    data = pd.read_csv(data_path, header=None)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values
    data = data.values

    # Most datasets have class as either the first or last column
    if class_index == "last":
        dataX = data[:, :-1]
        dataY = data[:, -1].astype(str)  # Class label is a string
    else:
        dataX = data[:, 1:]
        dataY = data[:, 0].astype(str)  # Class label is a string

    return pd.DataFrame(dataX), pd.DataFrame(dataY)