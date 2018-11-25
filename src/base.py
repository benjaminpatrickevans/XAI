from deap import gp, base, creator, tools, algorithms
from src import deapfix, search
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
import random
import numpy as np
import operator
from operator import eq
from functools import partial
import pygraphviz as pgv
import numbers
import os
import re
from sklearn import metrics

# The training data is just an ndarray
train_data_type = type("TrainData", (np.ndarray, ), {})

# A mask is an input record, which shows which path we must take through the tree. This is just a 1darray
mask_type = type("Mask", (np.ndarray, ), {})


class EvolutionaryBase(Classifier):

    def __init__(self, max_trees, max_depth, num_generations, verbose):
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.num_generations = num_generations
        self.verbose = verbose

        self._reset_pset()
        self.toolbox = self.create_toolbox(self.pset)

        self.crs_rate = 0.8
        self.mut_rate = 0.2

        # Cache for the trees
        self.evaluated_individuals = {}

    def _reset_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [mask_type, train_data_type], train_data_type)

        self.pset.renameArguments(ARG0="Mask", ARG1="TrainData")

    def create_toolbox(self, pset):
        # Maximising score
        creator.create('FitnessMax', base.Fitness, weights=(1.0, ))

        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", deapfix.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("select", tools.selDoubleTournament, fitness_size=7, parsimony_size=1.5, fitness_first=True)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", deapfix.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

        # Fitness function added in .fit as we need the data
        return toolbox

    def _categorical_feature_node(self, feature_idx, mask, *children):
        """
            A categorical node checks the output of its children, and
            returns the one which matches the specified mask.
        :param feature_idx:
        :param mask:
        :param children:
        :return:
        """
        children_outputs = [self._apply_filter(data, feature_idx, condition_check) for condition_check, data in children
                            if condition_check(mask[feature_idx])]

        if len(children_outputs) > 1:
            raise Exception("Multiple leaves true. They should be mutually exclusive")

        return children_outputs[0]

    def _numeric_feature_node(self, splitting_point, feature_idx, mask, *children):
        children_outputs = [self._apply_filter(data, feature_idx, partial(condition_check, splitting_point))
                            for condition_check, data in children
                            if condition_check(splitting_point, mask[feature_idx])]

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
        """
            Adds a categorical node into the tree. The categorical node takes
            each of the possible categories as children, meaning a branch is constructed
            for each category. No categorical binning is implemented.

        :param feature_values: The possible categories
        :param feature_index: The column index for the feature in the original data so we can access the values
        :param feature_name:
        :return:
        """

        # For strongly-typed GP we need to make a custom type for each value to preserve correct structure
        feature_node_input_types = []

        # Add the feature value nodes. These will form the children of the feature_node
        for value in feature_values:
            # TODO: Better regex?
            clean_value_name = re.sub("\.|-|,| ", "", str(value))  # Remove white space, full stops, commas and spaces
            feature_output_name = feature_name + "_" + clean_value_name + "Type"

            # Each feature value needs a special output type to ensure trees have a branch for each category
            feature_value_output_type = type(feature_output_name, (np.ndarray,), {})

            # Since we are in a for loop, must use val=value for the lambda. Check if the feature matches our value
            self.pset.addPrimitive(lambda data, val=value: (partial(eq, val), data),
                                   [train_data_type], feature_value_output_type,
                                   name=feature_name + "_" + clean_value_name)

            feature_node_input_types.append(feature_value_output_type)

        # Add the feature node, with the categorical inputs from above.
        self.pset.addPrimitive(lambda *xargs, feature_idx=feature_index: self._categorical_feature_node(feature_idx, *xargs),
                               [mask_type, *feature_node_input_types],
                               train_data_type, name="FN_"+feature_name)

    def _add_numeric_feature(self, feature_values, feature_index, feature_name):
        """
            Adds a binary splitting node for a numeric feature. The split criteria is just less than or
            equal to a a splitting point. The splitting point is a random value from the range of the feature.
            Less than or equal is used to split as this creates one branch for less than or equal, and another for
            greater than (i.e. a binary split).

        :param feature_values: The possible numeric values
        :param feature_index:  The column index for the feature in the original data so we can access the values
        :param feature_name:
        :return:
        """

        minimum_feature_value = min(feature_values)
        maximum_feature_value = max(feature_values)

        split_type = type(feature_name+"SplitPoint", (float, ), {})  # Splitting point type is just a restricted float

        # Terminals are random constant in the feature range
        self.pset.addEphemeralConstant(feature_name+"SplitPoint",
                                       lambda: random.uniform(minimum_feature_value, maximum_feature_value),
                                       split_type)

        # TODO: If integers should we add equals?
        operators = [operator.le, operator.gt]

        feature_node_input_types = []

        for split_operator in operators:

            feature_value_output_type = type(feature_name + "_" + split_operator.__name__ + "Type", (np.ndarray,), {})

            self.pset.addPrimitive(lambda data, op=split_operator: (op, data),
                                   [train_data_type], feature_value_output_type,
                                   name=feature_name + "_" + split_operator.__name__)

            feature_node_input_types.append(feature_value_output_type)

        self.pset.addPrimitive(lambda split, mask, *xargs, feature_idx=feature_index:
                               self._numeric_feature_node(split, feature_idx, mask, *xargs),
                               [split_type, mask_type, *feature_node_input_types],
                               train_data_type, name="FN_"+feature_name+split_operator.__name__)


    def _add_constructed_features(self, feature_indices):
        """
            Constructed features can be any combination of the original
            numeric features. i.e. f1 * f2 + f3. The splitting point is always <= 0.
            This allows many operators such as f1 < f2, i.e. by f1 - f2 <= 0?
            Random integer terminals can also used so the boundary is essentially flexible.
        :return:
        """

        # The constructed is just a column in an ndarray, we wrap this type for the constraints in STGP
        constructed_type = type("ConstructedFeature", (np.ndarray, ), {})

        def retrieve_feature(index):
            # Return a function which takes some data and gives you the data at the feature index
            return lambda data: data[:, index]

        for feature in feature_indices:
            # Note even though retrieve_feature doesnt actually return a constructed_type, we can treat it as such.
            # It actually returns a function which returns a constructed_type, this is what we do throughout since
            # the tree should be traversed from the root not from the leaves
            self.pset.addPrimitive(lambda ind=feature: retrieve_feature(ind), [], constructed_type,
                                   name="Feature"+str(feature))



        # Combination operators take 2 ndarrays and perform a mathematical function (i.e. add) on them.
        def divide(left, right):
            # Divide by zero return 0 instead of crashing
            return np.divide(left, right, out=np.zeros_like(left), where=right != 0)

        combination_operators = [np.add, np.multiply, np.subtract, divide]

        def combination_fn(l, r, op):
            # Return a function which takes in data and applies the operator to the children. Note this will
            # recurse when l and r contain combination operators themselves.
            return lambda data: op(l(data), r(data))

        # Add each of the combination operators
        for operation in combination_operators:
            self.pset.addPrimitive(lambda l, r, op=operation: combination_fn(l, r, op),
                                   [constructed_type, constructed_type],  # Takes in two values
                                   constructed_type,  # Outputs a single value
                                   name=operation.__name__)

        # We need to filter the constructed feature to be >=0, to do this we need to compare the new feature
        # with the original training data, and filter that data accordingly
        def constructed_feature(construct, mask, train):
            mask = mask.reshape(1, -1)  # Even though this is 1d, we want to treat as 2d so all operators can be uniform
            res = construct(mask)  # Construct the feature for the mask/input vector
            constructed = construct(train)  # Construct the feature for all the training data
            condition_met = constructed >= 0  # Retrieve the training rows where the constructed val is greater than 0

            # Now since the indices are the same, we can use this to access our original train data and filter
            # based on the mask. Once we have filtered the data, we no longer require the constructed feature.
            return train[np.where(condition_met)] if res >= 0 else train[np.where(~condition_met)]

        self.pset.addPrimitive(constructed_feature, [constructed_type, mask_type, train_data_type],
                               train_data_type,
                               name="ConstructedFilter")

    def _add_functions_and_terminals(self, x):

        num_instances, num_features = x.shape

        numeric_features = []

        for feature_index in range(num_features):
            feature_name = "Feature"+str(feature_index)
            # The unique values for the feature
            feature_values = set(x[:, feature_index])
            feature_type = type(next(iter(feature_values)))

            if feature_type == str:
                self._add_categorical_feature(feature_values, feature_index, feature_name)
            else:
                self._add_numeric_feature(feature_values, feature_index, feature_name)
                numeric_features.append(feature_index)

        self._add_constructed_features(numeric_features)

    def plot(self, file_name):
        if self.model is None:
            raise Exception("You must call fit before plot!")

        file_dir = "/".join(file_name.split("/")[:-1]) # Extract the directory from the file name

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        nodes, edges, labels = gp.graph(self.model)

        # For simpler graphs we want to ignore masks and training data
        ignore_indices = [idx for idx in labels if labels[idx] == "Mask" or labels[idx] == "TrainData"]

        # Remove the nodes and edges that point to any mask
        nodes = [node for node in nodes if node not in ignore_indices]
        edges = [edge for edge in edges if edge[0] not in ignore_indices and edge[1] not in ignore_indices]

        g = pgv.AGraph(outputorder="edgesfirst")
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            label = labels[i]

            if isinstance(label, numbers.Number):
                label = round(label, 2)

            label = str(label)

            # The child nodes indicate if a feature node was true or false
            if label.endswith("_le"):
                label = "False"
            elif label.endswith("_gt"):
                label = "True"
            elif label.startswith("FN_"):
                # Do some pretty formatting
                text = label.replace("FN_", "")
                text = text.replace("Feature", "Feature ")
                text = text.replace("gt", " > ")
                label = text

            n.attr["label"] = label
            n.attr["fillcolor"] = "white"
            n.attr["style"] = "filled"

        g.draw(file_name)

    def fit(self, x, y):
        # Ensure we use numpy arrays. Shuffle as well to be safe
        x, y = shuffle(np.asarray(x), np.asarray(y))

        num_instances, num_features = x.shape

        # Combine to make it easier for processing
        train_data = np.hstack((x, y))

        self.toolbox.register("evaluate", self._fitness_function, train_data=train_data)

        self._add_functions_and_terminals(x)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("mean", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        population_size = self.max_trees

        pop = self.toolbox.population(n=population_size)

        hof = tools.HallOfFame(1)  # We only use the best evolved model

        #population, logbook = search.eaMuPlusLambda(population=pop, toolbox=self.toolbox, mu=population_size,
        #                                            lambda_=population_size, cxpb=self.crs_rate, mutpb=self.mut_rate,
        #                                            ngen=self.num_generations, stats=stats, halloffame=hof)

        population, logbook = algorithms.eaSimple(pop, self.toolbox, self.crs_rate, self.mut_rate,
                                                  self.num_generations, stats=stats, halloffame=hof)

        # Debugging
        for model in population:
            print(model)

        # Now we want to use all the train data, so at test time the models are more robust
        self.train_data = train_data

        self.model = hof[0]
        self.models = population  # Final population