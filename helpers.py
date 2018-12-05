import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

def read_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values

    x = data.drop("class", axis=1).values
    y = 'class' + data["class"].astype(str)  # In case the class is just say "1", as h2o will try do regression

    y = np.reshape(y.values, (-1, 1))  # Flatten the y so its shape (len, 1)

    return x, y





class CategoricalToNumeric(TransformerMixin):
    '''
    Used for sklearn to convert categorical features to a onehotencoding.
    '''

    def fit(self, x, y=None):
        # Convert any categorical values to numeric
        data_x = pd.get_dummies(data=x)
        self.training_columns = data_x.columns # For recreating at test time
        return self

    def transform(self, x):
        '''
            http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/
        :param x:
        :return:
        '''

        x = pd.get_dummies(data=x)

        # In case we didnt see some of the examples in the test set
        missing_cols = set(self.training_columns) - set(x.columns)
        for c in missing_cols:
            x[c] = 0

        # Make sure we have all the columns we need
        assert (set(self.training_columns) - set(x.columns) == set())

        extra_cols = set(x.columns) - set(self.training_columns)

        if extra_cols:
            print("Extra columns in the unseen test data:", extra_cols)
            print("Ignoring them.")

        # Reorder to ensure we have the same columns and ordering as training data
        x = x[self.training_columns]

        return x
