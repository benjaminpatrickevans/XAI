from src.base import EvolutionaryBase
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import numpy as np


class GP(EvolutionaryBase):

    def __init__(self, max_trees=1024, max_depth=6, num_generations=50, verbose=0):
        super().__init__(max_trees, max_depth, num_generations, verbose)

    def _predict_for_instance(self, instance, training_data, tree, toolbox):
        class_probabilities, majority_class = self._predict_probabilities(instance, training_data, tree, toolbox)
        return majority_class

    def _predict_probabilities(self, instance, training_data, tree, toolbox):
        callable_tree = toolbox.compile(expr=tree)
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

        # Return the probabilities and the majority class
        return class_probabilities, classes[majority_class_idx]

    def _fitness_function(self, individual, train_data):
        kf = KFold(random_state=0)
        
        scores = []

        # We output a failure string for computing diversity. Since diversity is related to the entire population,
        # this can only be done AFTER computing all the fitness values. This is done in search.diversity_search
        failure_vector = []

        for train_index, test_index in kf.split(train_data):
            training_data, valid_data = train_data[train_index], train_data[test_index]
            real_labels = valid_data[:, -1]

            predictions = [self._predict_for_instance(instance, training_data, individual, self.toolbox)
                           for instance in valid_data]

            f1 = metrics.f1_score(real_labels, predictions, average="weighted")
            scores.append(f1)

        score = np.mean(scores)

        if self.verbose:
            print(individual, score)

        return score,

    def predict(self, x):
        if self.model is None:
            raise Exception("You must call fit before predict!")

        x = np.asarray(x)
        return [self._predict_for_instance(instance, self.train_data, self.model, self.toolbox)
                for instance in x]
