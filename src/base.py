from deap import gp, algorithms, base, creator, tools
from src import deapfix
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
import numpy as np
from operator import eq
from functools import partial
import operator
import itertools
import random

class EvolutionaryForest(Classifier):

    def __init__(self, max_trees=10, max_depth=10, num_generations=50):
        self.max_trees = max_trees
        self.tree_depth = max_depth
        self.num_generations = num_generations

        self._reset_pset()
        self.toolbox = self.create_toolbox(self.pset)

        self.crs_rate = 0.7
        self.mut_rate = 0.25

    def _reset_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray], np.ndarray)

    def _fitness_function(self, individual, data_x, data_y):
        func = self.toolbox.compile(expr=individual)
        matching_data = func(data_x)
        print(individual)
        print(matching_data.shape)
        return 0,

    def create_toolbox(self, pset):
        toolbox = base.Toolbox()

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
            featue_node_input_types = []

            # Add the feature value nodes. These will form the children of the feature_node
            for value in feature_values:

                feature_output_name = feature_name+"_"+str(value) + "Type"

                # Each feature value needs a special output type to ensure trees have a branch for each category
                feature_value_output_type = type(feature_output_name, (np.ndarray,), {})

                # Since we are in a for loop, must use val=value for the lambda. Check if the feature matches our value
                self.pset.addPrimitive(lambda data, val=value, feature_idx=feature_index:
                                       self.apply_filter(data, feature_idx, partial(eq, val)),
                                       [np.ndarray], feature_value_output_type,
                                       name=feature_name+"_"+str(value))

                featue_node_input_types.append(feature_value_output_type)

            # Add the feature node, with the categorical inputs from above.
            self.pset.addPrimitive(self.feature_node, [*featue_node_input_types], np.ndarray, name=feature_name)

    def _add_terminal_set(self, x):
        self.pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)

        self.pset.addTerminal(False, bool)
        self.pset.addTerminal(True, bool)

    def fit(self, x, y):
        # Ensure we use numpy arrays
        x, y = shuffle(np.asarray(x), np.asarray(y))

        self.toolbox.register("evaluate", self._fitness_function, data_x=x, data_y=y)

        self._add_function_set(x)
        self._add_terminal_set(x)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = self.toolbox.population(n=self.max_trees)

        algorithms.eaSimple(pop, self.toolbox, self.crs_rate, self.mut_rate, self.num_generations, stats=stats)