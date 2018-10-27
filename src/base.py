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

# The training data is just an ndarray
train_data_type = type("TrainData", (np.ndarray, ), {})

# A mask is an input record, which shows which path we must take through the tree. This is just a 1darray
mask_type = type("Mask", (np.ndarray, ), {})

class EvolutionaryBase(Classifier):

    def __init__(self, max_trees=10, max_depth=10, num_generations=50):
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.num_generations = num_generations

        self._reset_pset()
        self.toolbox = self.create_toolbox(self.pset)

        self.crs_rate = 0.7
        self.mut_rate = 0.25

    def _reset_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [mask_type, train_data_type], train_data_type)

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

    def create_toolbox(self, pset):
        # Fitness is f1-score. So the larger the better
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                       pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", deapfix.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", deapfix.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # Fitness function added in .fit as we need the data

        return toolbox

    def feature_node(self, feature_idx, mask, *children):
        """
            A feature node checks the output of its children, and
            returns the one which matches the specified mask.

        :param children:
        :return:
        """

        children_outputs = [self.apply_filter(data, feature_idx, condition_check) for condition_check, data in children
                            if condition_check(mask[feature_idx])]

        if len(children_outputs) > 1:
            raise Exception("Multiple leaves true. They should be mutually exclusive")

        return children_outputs[0]

    def apply_filter(self, data, feature_index, condition):

        # Apply the condition to the feature_index, returning the indices of rows where the condition was met
        filtered_indices = np.where(condition(data[:, feature_index]))

        # Extract only the rows where the condition was true
        filtered_data = data[filtered_indices]

        return filtered_data

    def _add_functions_and_terminals(self, x):

        num_instances, num_features = x.shape

        for feature_index in range(num_features):
            feature_name = "Feature"+str(feature_index)
            # The unique values for the feature
            feature_values = set(x[:, feature_index])

            # For strongly-typed GP we need to make a custom type for each value to preserve correct structure
            feature_node_input_types = []

            # Add the feature value nodes. These will form the children of the feature_node
            for value in feature_values:

                print("Value is: ", value)

                feature_output_name = feature_name+"_"+str(value) + "Type"

                # Each feature value needs a special output type to ensure trees have a branch for each category
                feature_value_output_type = type(feature_output_name, (np.ndarray,), {})

                # Since we are in a for loop, must use val=value for the lambda. Check if the feature matches our value
                self.pset.addPrimitive(lambda data, val=value: (partial(eq, val), data),
                                       [train_data_type], feature_value_output_type,
                                       name=feature_name+"_"+str(value))

                feature_node_input_types.append(feature_value_output_type)

            # Add the feature node, with the categorical inputs from above.
            self.pset.addPrimitive(lambda *xargs, feature_idx=feature_index: self.feature_node(feature_idx, *xargs),
                                   [mask_type, *feature_node_input_types],
                                   train_data_type, name=feature_name)



    def fit(self, x, y):
        # Ensure we use numpy arrays
        x, y = shuffle(np.asarray(x), np.asarray(y))

        num_instances, num_features = x.shape

        # We should never exceed the user specified height of trees, however, if this height is too large we should
        # limit it. If a tree height is > num features, by definition duplicates must appear which we want to
        # minimise

        print("Max depth was:", self.max_depth)
        self.max_depth = min(self.max_depth, num_features)
        print("Max depth changed to:", self.max_depth)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

        # Combine to make it easier for processing
        train_data = np.hstack((x, y))

        self.toolbox.register("evaluate", self._fitness_function, train_data=train_data)

        self._add_functions_and_terminals(x)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        pop = self.toolbox.population(n=self.max_trees)

        algorithms.eaSimple(pop, self.toolbox, self.crs_rate, self.mut_rate, self.num_generations, stats=stats)