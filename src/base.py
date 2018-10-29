from deap import gp, algorithms, base, creator, tools
from src import deapfix
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
import random
import numpy as np
import operator
from operator import eq
from functools import partial

# The training data is just an ndarray
train_data_type = type("TrainData", (np.ndarray, ), {})

# A mask is an input record, which shows which path we must take through the tree. This is just a 1darray
mask_type = type("Mask", (np.ndarray, ), {})


class EvolutionaryBase(Classifier):

    def __init__(self, max_trees, max_depth, num_generations):
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.num_generations = num_generations

        self._reset_pset()
        self.toolbox = self.create_toolbox(self.pset)

        self.crs_rate = 0.7
        self.mut_rate = 0.25

    def _reset_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [mask_type, train_data_type], train_data_type)

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

    def categorical_feature_node(self, feature_idx, mask, *children):
        """
            A feature node checks the output of its children, and
            returns the one which matches the specified mask.

        :param children:
        :return:
        """

        children_outputs = [self._apply_filter(data, feature_idx, condition_check) for condition_check, data in children
                            if condition_check(mask[feature_idx])]

        if len(children_outputs) > 1:
            raise Exception("Multiple leaves true. They should be mutually exclusive")

        return children_outputs[0]

    def numeric_feature_node(self, split, feature_idx, mask, *children):
        children_outputs = [self._apply_filter(data, feature_idx, partial(condition_check, split))
                            for condition_check, data in children
                            if condition_check(split, mask[feature_idx])]

        if len(children_outputs) > 1:
            raise Exception("Multiple leaves true. They should be mutually exclusive")

        return children_outputs[0]


    def _apply_filter(self, data, feature_index, condition):

        # Apply the condition to the feature_index, returning the indices of rows where the condition was met
        filtered_indices = np.where(condition(data[:, feature_index]))

        # Extract only the rows where the condition was true
        filtered_data = data[filtered_indices]

        return filtered_data

    def _add_categorical_feature(self, feature_values, feature_index, feature_name):

        # For strongly-typed GP we need to make a custom type for each value to preserve correct structure
        feature_node_input_types = []

        # Add the feature value nodes. These will form the children of the feature_node
        for value in feature_values:
            feature_output_name = feature_name + "_" + str(value) + "Type"

            # Each feature value needs a special output type to ensure trees have a branch for each category
            feature_value_output_type = type(feature_output_name, (np.ndarray,), {})

            # Since we are in a for loop, must use val=value for the lambda. Check if the feature matches our value
            self.pset.addPrimitive(lambda data, val=value: (partial(eq, val), data),
                                   [train_data_type], feature_value_output_type,
                                   name=feature_name + "_" + str(value))

            feature_node_input_types.append(feature_value_output_type)

        # Add the feature node, with the categorical inputs from above.
        self.pset.addPrimitive(lambda *xargs, feature_idx=feature_index: self.categorical_feature_node(feature_idx, *xargs),
                               [mask_type, *feature_node_input_types],
                               train_data_type, name=feature_name)

    def _add_numeric_feature(self, feature_values, feature_index, feature_name):

        minimum_feature_value = min(feature_values)
        maximum_feature_value = max(feature_values)

        split_type = type(feature_name+"Split", (float, ), {})  # Splitting point type is just a restricted float

        # Terminals are random constant in the feature range
        self.pset.addEphemeralConstant(feature_name+"Split",
                                       lambda: random.uniform(minimum_feature_value, maximum_feature_value),
                                       split_type)

        # TODO: If integers should we add equals?
        operators = [operator.le, operator.gt]

        feature_node_input_types = []

        for split_operator in operators:

            feature_value_output_type = type(feature_name + "_" + split_operator.__name__ + "Type", (np.ndarray,), {})

            self.pset.addPrimitive(lambda data, op=split_operator: (op, data),
                                   [train_data_type], feature_value_output_type,
                                   name=feature_name + "_" + split_operator.__name__ + split_operator.__name__)

            feature_node_input_types.append(feature_value_output_type)

        self.pset.addPrimitive(lambda split, mask, *xargs, feature_idx=feature_index:
                               self.numeric_feature_node(split, feature_idx, mask, *xargs),
                               [split_type, mask_type, *feature_node_input_types],
                               train_data_type, name=feature_name+split_operator.__name__)


    def _add_functions_and_terminals(self, x):

        num_instances, num_features = x.shape

        for feature_index in range(num_features):
            feature_name = "Feature"+str(feature_index)
            # The unique values for the feature
            feature_values = set(x[:, feature_index])
            feature_type = type(next(iter(feature_values)))

            print(feature_values, feature_type)

            if feature_type == str:
                self._add_categorical_feature(feature_values, feature_index, feature_name)
            else:
                self._add_numeric_feature(feature_values, feature_index, feature_name)


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

        hof = tools.HallOfFame(1)  # We only use the best evolved model

        algorithms.eaSimple(pop, self.toolbox, self.crs_rate, self.mut_rate, self.num_generations, stats=stats,
                            halloffame=hof)

        self.model = hof[0]

        # Temporary
        self.train_data = train_data