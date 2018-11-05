from deap import tools, algorithms
import itertools
import numpy as np
from scipy.stats import mode

def _hamming(a, b):
    assert a.shape == b.shape
    return np.count_nonzero(a != b)

def diversity_search(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None,
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

        all_failure_vectors = []

        # To begin with we can only set the f1_score
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0], 0 # Set the diversity to 0 as we do not know it yet
            ind.failure_vector = fit[1]  # Extract the failure vector
            all_failure_vectors.append(ind.failure_vector)

        all_failure_vectors = np.asarray(all_failure_vectors, dtype=bool)

        # Since the ensemble uses majority voting, compute the most common success/failure for each instance
        ensemble_failures = mode(all_failure_vectors)[0].flatten()
        num_ensemble_failures = np.count_nonzero(ensemble_failures == 0)

        # Now we must compute the diversity for all individuals. We must do this for all individuals
        # not just the invalid ones, as the diversity changes based on the entire population. Check
        # the paper referenced in the doc string for the related formulas/explanations

        for ind in offspring:
            failure_vector = ind.failure_vector
            hamming_distance = _hamming(failure_vector, ensemble_failures)
            sum_num_failures = np.count_nonzero(failure_vector == 0) + num_ensemble_failures

            # Protect against divide by zero
            pfc = (hamming_distance / sum_num_failures) if sum_num_failures > 0 else 1

            # Keep the original f1 score, but set the new diversity fitness (the pfc)
            ind.fitness.values = ind.fitness.values[0], pfc

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

        if num_ensemble_failures == 0:
            print("Ensemble classified every instance correctly, exiting early!")
            break

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

    return population, logbook
