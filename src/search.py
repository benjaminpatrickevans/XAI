from deap import tools, algorithms, gp
import numpy as np
from copy import copy

def _hamming(a, b):
    assert a.shape == b.shape
    return np.count_nonzero(a != b)

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None,
                   verbose=__debug__):
    """This is a modification of the :math:`(\mu + \lambda)` evolutionary algorithm.
    The original algorithm can be found at deap.algorithms.eaMuPlusLambda.

    This has been modified to compute the pairwise failure credit (see:
    "Ensemble learning using multi-objective evolutionary algorithms" 2006) as
    the secondary objective.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Off spring is jsut the population to begin with
    offspring = population

    # Begin the generational process
    for gen in range(0, ngen + 1):

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        # To begin with we can only set the f1_score
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Prune
        offspring = [prune(ind) for ind in offspring]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

    return population, logbook


def prune(ind):
    # Just to be safe, copy the tree
    tree = copy(ind)

    # If we cant prune anymore, return the tree.
    if not tree:
        print("Unable to prune any further", tree)
        return tree

    children_outputs = []

    # Start at 1 as searchSubtree(0) is self
    for child in range(1, len(tree)):
        # Slice of the original tree where the new tree is present
        subtree_slice = tree.searchSubtree(child)

        # Convert to a tree so we can recurse
        subtree = gp.PrimitiveTree(tree[subtree_slice])

        # Recurse on the subtree
        tree[subtree_slice] = prune(subtree)

        # Call with the input
        #output = tree[subtree_slice]

    return tree
