from deap import gp
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
import numpy as np
from operator import eq
from functools import partial

class EvolutionaryForest(Classifier):

    def __init__(self, max_trees=10, max_depth=10, num_generations=50):
        self.max_trees = max_trees
        self.tree_depth = max_depth
        self.num_generations = num_generations

        self._reset_pset()

    def _reset_pset(self):
        self.pset = gp.PrimitiveSetTyped("main", [np.ndarray], np.ndarray)

    def feature_node(self, data, *args):
        """ A feature node doesnt actually need to do anything, as all the processing is done
        in the children nodes. This just serves as a dummy node to make the trees clearer, and
        allow better/valid breeding as we can crossover feature nodes."""

        return data

    def apply_filter(self, data, feature_index, condition):

        # Apply the condition to the feature_index, returning the indices of rows where the condition was met
        filtered_indices = np.where(condition(data[:, feature_index]))

        # Extract only the rows where the condition was true
        filtered_data = data[filtered_indices]

        return filtered_data

    def _add_function_set(self, x):

        num_instances, num_features = x.shape

        for feature_index in range(num_features):
            feature_name = "Feature"+str(feature_index)
            # The unique values for the feature
            feature_values = set(x[:, feature_index])

            # For strongly-typed GP we need to make a custom type for each value to preserve correct structure
            input_types = []

            # Add the feature value nodes. These will form the children of the feature_node
            for value in feature_values:

                feature_output_name = feature_name+"_"+str(value) + "Type"
                feature_output_type = type(feature_output_name, (np.ndarray,), {})

                # Since we are in a for loop, must use val=value for the lambda. Check if the feature matches our value
                self.pset.addPrimitive(lambda data, val=value:
                                       self.apply_filter(data, feature_index, partial(eq, val)),
                                       [np.ndarray], feature_output_type, # Takes in a ndarray and returns special type
                                       name=feature_name+"_"+str(value))

                input_types.append(feature_output_type)

                # Todo: do we need this?
                self.pset.context[feature_output_name] = feature_output_type

            # Add the feature node, with the categorical inputs from above
            self.pset.addPrimitive(self.feature_node, [*input_types], np.ndarray, name=feature_name)

    def _add_terminal_set(self, x):
        pass

    def fit(self, x, y):
        # Ensure we use numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        x, y = shuffle(x, y)

        self._add_function_set(x)


        pass