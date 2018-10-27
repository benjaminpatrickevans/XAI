from src.base import EvolutionaryBase
from deap import gp, algorithms, base, creator, tools
from src import deapfix
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import operator
from operator import eq
from functools import partial

class EvolutionaryForest(EvolutionaryBase):

    def __init__(self, max_trees=10, max_depth=10, num_generations=50):
        super().__init__(max_trees, max_depth, num_generations)

    def _get_majority_class(self, labels):
        classes, class_counts = np.unique(labels, return_counts=True)
        class_probabilities = class_counts / np.sum(class_counts)  # Normalize the counts

        majority_class_idx = np.argmax(class_probabilities)
        return classes[majority_class_idx]

    def _fitness_function(self, individual, train_data):
        callable_tree = self.toolbox.compile(expr=individual)

        training_data, valid_data = train_test_split(train_data, test_size=0.2)

        majority_class = self._get_majority_class(training_data[:, -1])

        predictions = []

        for mask in valid_data:

            # The subset of data that matches our mask
            matching_data = callable_tree(mask, training_data)

            if matching_data is None or matching_data.shape[0] == 0:
                # Predict majority class, this means either the tree is bad or we have never seen an instance like this
                prediction = majority_class
            else:
                # Compute the most common class from only the resulting data
                prediction = self._get_majority_class(matching_data[:, -1])

            predictions.append(prediction)

        f1 = metrics.f1_score(valid_data[:, -1], predictions, average="weighted")

        print(individual, f1)
        return f1,