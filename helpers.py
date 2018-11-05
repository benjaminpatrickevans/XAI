import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def read_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values

    x = data.drop("class", axis=1).values
    y = 'class' + data["class"].astype(str)  # In case the class is just say "1", as h2o will try do regression

    y = np.reshape(y.values, (-1, 1))  # Flatten the y so its shape (len, 1)

    return x, y


def evaluate(name, preds, real):
    score = f1_score(real, preds, average="weighted")
    print(name, "%.3f" % score)
    return score
