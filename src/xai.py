from src.base import EvolutionaryBase
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np


class GP(EvolutionaryBase):

    def __init__(self, max_trees=1024, max_depth=10, num_generations=50, verbose=0):
        super().__init__(max_trees, max_depth, num_generations, verbose)

    def _predict_for_instance(self, instance, training_data, tree, toolbox):
        callable_tree = toolbox.compile(expr=tree)

        matching_data = callable_tree(instance, training_data)

        # Using the matching data, if there was none then just use the training set
        labels = matching_data[:, -1] if matching_data.size else training_data[:, -1]

        # Count the occurences for each class
        classes, class_counts = np.unique(labels, return_counts=True)

        # Find the class that achieved the most counts
        most_common_class_idx = np.argmax(class_counts)

        # Return this class
        return classes[most_common_class_idx]

    def complexity(self, individual):
        # Number of features in the tree
        num_nodes = sum(1 if node.name.startswith("FN_") or node.name.startswith("CFN_") else 0 for node in individual)

        # Note: Technically with constructed features this could be higher than 1
        complexity = num_nodes / self.max_depth

        # We don't want to encourage trees with no features!
        if complexity == 0:
            complexity = 1

        # Cap the complexity at 1
        return min(1, complexity)

    def _fitness_function(self, individual, train_data):
        tree_str = str(individual)

        # Avoid recomputing fitness
        if tree_str in self.cache:
            return self.cache[tree_str]

        kf = KFold(random_state=0)
        scores = []

        for train_index, test_index in kf.split(train_data):
            training_data, valid_data = train_data[train_index], train_data[test_index]
            real_labels = valid_data[:, -1]

            predictions = [self._predict_for_instance(instance, training_data, individual, self.toolbox)
                           for instance in valid_data]

            f1 = metrics.f1_score(real_labels, predictions, average="weighted")
            scores.append(f1)

        average_score = np.mean(scores)
        complexity = self.complexity(individual)

        fitness = average_score, complexity

        # Store in cache so we don't need to reevaluate
        self.cache[tree_str] = fitness

        if self.verbose:
            print(individual, fitness)

        return fitness

    def predict(self, x):
        if self.model is None:
            raise Exception("You must call fit before predict!")

        # Ensure x is a ndarray
        x = np.asarray(x)

        return [self._predict_for_instance(instance, self.train_data, self.model, self.toolbox)
                for instance in x]
