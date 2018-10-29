from src.base import EvolutionaryBase
from deap import gp, algorithms, base, creator, tools
from src import deapfix
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


class EvolutionaryForest(EvolutionaryBase):

    def __init__(self, max_trees=10, max_depth=10, num_generations=50, verbose=0):
        super().__init__(max_trees, max_depth, num_generations, verbose)

    def _predict_for_instance(self, instance, training_data, individual):
        probabilities, majority_class = self._predict_probabilities(individual, instance, training_data)
        return majority_class

    def _predict_probabilities(self, tree, instance, training_data):
        callable_tree = self.toolbox.compile(expr=tree)
        matching_data = callable_tree(instance, training_data)

        if matching_data is None or matching_data.shape[0] == 0:
            # If there's no matching data, use the entire training set
            labels = training_data[:, -1]
        else:
            # Else its a valid tree so use the matching data
            labels = matching_data[:, -1]

        classes, class_counts = np.unique(labels, return_counts=True)

        # It could be the case that a particular class never occurred at this path, so we must assign a zero count
        missing_classes = set(training_data[:, -1]) - set(classes)

        if missing_classes:
            # Add them in
            classes = np.append(classes, list(missing_classes))
            class_counts = np.append(class_counts, [0] * len(missing_classes))

            # Once classes have been added, these need to be sorted as the order is important when averaging
            sorted_indices = classes.argsort()
            classes = classes[sorted_indices]
            class_counts = class_counts[sorted_indices]

        class_probabilities = class_counts / np.sum(class_counts)  # Normalize the counts

        majority_class_idx = np.argmax(class_probabilities)

        return class_probabilities, classes[majority_class_idx]

    def _fitness_function(self, individual, train_data):
        training_data, valid_data = train_test_split(train_data, test_size=0.2)
        predictions = [self._predict_for_instance(instance, training_data, individual) for instance in valid_data]
        f1 = metrics.f1_score(valid_data[:, -1], predictions, average="weighted")

        if self.verbose:
            print(individual, f1)

        return f1,

    def predict(self, x):
        if self.model is None:
            raise Exception("You must call fit before predict!")

        x = np.asarray(x)
        return [self._predict_for_instance(instance, self.train_data, self.model) for instance in x]


    def predict_majority(self, x):
        if self.models is None:
            raise Exception("You must call fit before predict!")

        return self._soft_voting(x)

    def _soft_voting(self, x, weights=None):

        x = np.asarray(x)
        predictions = []

        all_classes = np.unique(self.train_data[:, -1])

        for instance in x:
            class_probabilities = [self._predict_probabilities(model, instance, self.train_data)[0]
                                   for model in self.models] # The predicted probability vector from each model

            class_probabilities = np.average(class_probabilities, axis=0)  # Average across the models

            prediction = all_classes[np.argmax(class_probabilities)]  # Choose the class with highest average

            predictions.append(prediction)

        return predictions


