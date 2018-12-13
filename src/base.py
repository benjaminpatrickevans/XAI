from deap import gp, base, creator, tools, algorithms
from src import deapcustom, plotter
from sklearn.base import ClassifierMixin as Classifier
from sklearn.utils import shuffle
import random
import numpy as np
import operator
from operator import eq
from functools import partial
import re

# The training data is just an ndarray
train_data_type = type("TrainData", (np.ndarray, ), {})

# A mask is an input record, which shows which path we must take through the tree. This is just a 1darray
mask_type = type("Mask", (np.ndarray, ), {})


class EvolutionaryBase(Classifier):

    def __init__(self, max_trees, max_depth, num_generations, verbose):
        self.num_trees = max_trees
        self.max_depth = max_depth
        self.num_generations = num_generations
        self.verbose = verbose

        self.crs_rate = 0.8
        self.mut_rate = 0.2

        # Cache for the trees
        self.cache = {}

        self.pset = self.create_pset()
        self.toolbox = self.create_toolbox(self.pset)

    def create_pset(self):
        pset = gp.PrimitiveSetTyped("MAIN", [mask_type, train_data_type], train_data_type)
        pset.renameArguments(ARG0="Mask", ARG1="TrainData")
        return pset

    def create_toolbox(self, pset):
        # Maximising score, minimising complexity
        creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))

        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        toolbox.register("expr", deapcustom.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("select", tools.selNSGA2, nd="log")
        toolbox.register("mate", deapcustom.repeated_crossover, existing=self.cache, toolbox=toolbox)
        toolbox.register("expr_mut", deapcustom.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutate", deapcustom.repeated_mutation, expr=toolbox.expr_mut, pset=pset, existing=self.cache,
                         toolbox=toolbox)

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
        return data[filtered_indices]

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
        for category in feature_values:
            # TODO: Better regex?
            clean_category_name = re.sub("\.|-|,| ", "", str(category))  # Remove white space, full stops, commas and spaces
            feature_output_name = feature_name + "_" + clean_category_name + "Type"

            # Each feature value needs a special output type to ensure trees have a branch for each category
            feature_value_output_type = type(feature_output_name, (np.ndarray,), {})

            # Since we are in a for loop, must use val=value for the lambda. Check if the feature matches our value
            self.pset.addPrimitive(lambda data, val=category: (partial(eq, val), data),
                                   [train_data_type], feature_value_output_type,
                                   name=feature_name + "_category" + clean_category_name)

            feature_node_input_types.append(feature_value_output_type)

        # Add the feature node, with the categorical inputs from above.
        self.pset.addPrimitive(lambda *xargs, feature_idx=feature_index: self._categorical_feature_node(feature_idx, *xargs),
                               [mask_type, *feature_node_input_types],
                               train_data_type, name="FN_Category"+feature_name)

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
                               train_data_type, name="FN_Numeric"+feature_name)


    def _add_constructed_features(self, feature_indices):
        """
            Constructed features can be any combination of the original
            numeric features. i.e. f1 * f2 + f3. The splitting point is always <= 0.
            This allows many operators such as f1 < f2, i.e. by f1 - f2 <= 0?
            Random integer terminals can also used so the boundary is essentially flexible.
        :return:
        """

        # The constructed is just a column in an ndarray, we wrap this type for the constraints in STGP
        constructed_type = type("ConstructedFeature", (np.ndarray,), {})

        def retrieve_feature(index):
            # Return a function which takes some data and gives you the data at the feature index
            return lambda data: data[:, index]

        for feature in feature_indices:
            # Note even though retrieve_feature doesnt actually return a constructed_type, we can treat it as such.
            # It actually returns a function which returns a constructed_type, this is what we do throughout since
            # the tree should be traversed from the root not from the leaves
            self.pset.addPrimitive(lambda ind=feature: retrieve_feature(ind), [], constructed_type,
                                   name="CFN_Feature" + str(feature))

        def divide(left, right):
            # Divide by zero return 0 instead of crashing
            return np.divide(left, right, out=np.zeros_like(left), where=right != 0)

        # Combination operators take 2 ndarrays and perform a mathematical function (i.e. add) on them.
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




        # We need to filter the constructed feature to be > 0, to do this we need to compare the new feature
        # with the mask, and return the appropriate branch.
        def constructed_feature(construct, mask, false_branch, true_branch):
            mask = mask.reshape(1, -1)  # Even though this is 1d, we want to treat as 2d so all operators can be uniform
            res = construct(mask)  # Construct the feature for the mask/input vector

            # Filter based on whether the constructed mask feature was >= 0 or not
            return true_branch[np.where(construct(true_branch) > 0)] if res >= 0\
                else false_branch[np.where(construct(false_branch) <= 0)]

        self.pset.addPrimitive(constructed_feature,
                               [constructed_type, mask_type, train_data_type, train_data_type],
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

            # Treat numeric features as categorical if they're only 1/0 as well
            if feature_type == str or len(feature_values) == 2:
                self._add_categorical_feature(feature_values, feature_index, feature_name)
            else:
                self._add_numeric_feature(feature_values, feature_index, feature_name)
                numeric_features.append(feature_index)

        if numeric_features:
            self._add_constructed_features(numeric_features)

    def plot(self, file_name):
        if self.model is None:
            raise Exception("You must call fit before plot!")

        plotter.plot_model(self.model, file_name, self.train_data)

    def plot_pareto(self, file_name):
        if self.pareto_front is None:
            raise Exception("You must call fit before plotting!")

        plotter.plot_pareto(self.pareto_front, self.final_population, file_name)

    def fit(self, x, y):
        # Ensure we use numpy arrays. Shuffle as well to be safe
        x, y = shuffle(np.asarray(x), np.asarray(y))

        # Combine to make it easier for processing
        train_data = np.hstack((x, y))

        self.toolbox.register("evaluate", self._fitness_function, train_data=train_data)

        self._add_functions_and_terminals(x)

        def pretty_format(fn, out, axis=0):
            # Format the two objectives by rounding to 3dp
            return np.round(fn(out, axis=axis), 3)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", partial(pretty_format, np.min))
        stats.register("mean", partial(pretty_format, np.mean))
        stats.register("max", partial(pretty_format, np.max))
        stats.register("std", partial(pretty_format, np.std))

        population_size = self.num_trees

        pop = self.toolbox.population(n=population_size)

        def rough_eq(ind1, ind2):
            # If the 2 fitnesses are very close
            return np.allclose(ind1.fitness.values, ind2.fitness.values)

        hof = tools.ParetoFront(similar=rough_eq)

        population, logbook = algorithms.eaMuPlusLambda(population=pop, toolbox=self.toolbox, mu=population_size,
                                                    lambda_=population_size, cxpb=self.crs_rate, mutpb=self.mut_rate,
                                                    ngen=self.num_generations, stats=stats, halloffame=hof)

        if self.verbose:
            print("Best model found:", hof[0])
            print("Pareto front", [ind.fitness.values for ind in hof])

        # For access at test time. TODO: Can we store the probabilities rather than the train data for speed?
        self.train_data = train_data

        # Store the entire pareto front
        self.pareto_front = hof

        # Store the final population (for plotting)
        self.final_population = population

        # The model with highest score is pareto_front[0]
        self.model = self.pareto_front[0]

        print("Percentage of unique models: %.2f%%" % (len(self.cache) / (self.num_generations * self.num_trees) * 100))

        # Clear cache
        self.cache = {}