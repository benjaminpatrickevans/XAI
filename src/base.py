from deap import gp
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
import numpy as np


class EvolutionaryForest(Classifier):

    def __init__(self, max_trees=10, max_depth=10, num_generations=50):
        self.max_trees = max_trees
        self.tree_depth = max_depth
        self.num_generations = num_generations

        self._reset_pset()

    def _reset_pset(self):
        self.pset = gp.PrimitiveSetTyped("main", [bool, float], np.array)

    def apply_filter(self, data, feature_index, condition):

        # Apply the condition to the feature_index, returning the indices of rows where the condition was met
        filtered_indices = np.where(condition(data[:, feature_index]))

        # Extract only the rows where the condition was true
        filtered_data = data[filtered_indices]

        return filtered_data

    def _add_function_set(self, x):

        num_instances, num_features = x.shape

        for feature in range(num_features):
            # The unique values for the feature
            feature_values = set(x[:, feature])


            print("Need to add a node for feature", feature, "with a child for", feature_values)

            # Need to add each feature value as a terminal

            # Then need to add the feature as a primitive/function
            #self.pset.addPrimitive(primitive, in_types, np.array, name="Feature"+str(feature))


        print(x.shape)
        pass

    def _add_terminal_set(self, x):
        pass

    def fit(self, x, y):
        # Ensure we use numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        x, y = shuffle(x, y)

        self._add_function_set(x)


        pass