import random
from inspect import isclass
from deap import gp

def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth >= height
    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth >= height or \
            (depth >= min_ and random.random() < pset.terminalRatio)
    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)

def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    This differs from the original generate function in
    three ways. If we try add a terminal and none is available, it
    adds a primtive instead. Likewise, some nodes are only available as terminals
    so do not try add primitives in these cases. Finally, each tree
    should only have one of each feature node type. So this is ensured
    while generating new trees.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()

        # We should add a terminal if we are at the desired height or alternatively if we are at a node that is
        # only available as a terminal
        if condition(height, depth) or type_.__name__ == "Mask" or "SplitPoint" in type_.__name__:
            try:
                term = random.choice(pset.terminals[type_])

                if isclass(term):
                    term = term()
                expr.append(term)
            except IndexError:
                # If we try add a terminal but none are found, add a primitive instead
                force_prefix = None

                # If we try add a constructed feature but we are at the termination criteria, we should
                # only add a feature, not another combiner node!
                if type_.__name__ == "ConstructedFeature":
                    force_prefix = "CFN_Feature"

                # Add a primitive, giving the optional type (force_prefix)
                add_primitive(pset, type_, expr, stack, depth, force_prefix)

        else:
            success = add_primitive(pset, type_, expr, stack, depth)

            if not success:
                # Then just end the tree creation by setting the depth to be the max height
                stack.append((height, type_))

    return expr


def add_primitive(pset, type_, expr, stack, depth, force_prefix=None):
    all_nodes = pset.primitives[type_]

    # We need to ensure each feature node only occurs once
    existing_nodes = set([node.name for node in expr if node.name.startswith("FN_")])

    # Exclude the nodes that already exist in the expr
    options = [node for node in all_nodes if node.name not in existing_nodes]

    if force_prefix:
        # Then we need to exclude the other options
        options = [node for node in options if node.name.startswith(force_prefix)]

    if not options:
        return False

    prim = random.choice(options)

    expr.append(prim)
    for arg in reversed(prim.args):
        stack.append((depth + 1, arg))

    return True


def repeated_mutation(individual, expr, pset, existing, toolbox, max_tries=10):
    """
        Repeated apply mutUniform until the mutated individual has
        not existed before.
    :param individual:
    :param expr:
    :param pset:
    :return:
    """

    # Try for max_tries, or until we generate a unique individual
    for i in range(max_tries):
        ind = toolbox.clone(individual)

        mutated = gp.mutUniform(ind, expr, pset)

        # mutUniform returns a tuple, so access the first element of the tuple and see if that is unique
        if str(mutated[0]) not in existing:
            break

    return mutated


def repeated_crossover(ind1, ind2, existing, toolbox, max_tries=10):
    """
        Repeatedly apply cxOnePoint until the generated individuals are
        unique from the existing originals (or until max_tries is hit).
        Thiw was inspired by tpots _mate_operator.
    :param ind1:
    :param ind2:
    :param existing:
    :param toolbox:
    :param max_tries:
    :return:
    """
    unique_offspring1 = None
    unique_offspring2 = None

    # Try for max_tries, or until we generate a unique individual
    for i in range(max_tries):
        ind1_copy, ind2_copy = toolbox.clone(ind1), toolbox.clone(ind2)

        offspring1, offspring2 = gp.cxOnePoint(ind1_copy, ind2_copy)

        if str(offspring1) not in existing:
            unique_offspring1 = offspring1

        if str(offspring2) not in existing:
            unique_offspring2 = offspring2

        # Only break once both are unique
        if unique_offspring1 and unique_offspring2:
            break

    # If we didnt find a unique, then use the last (repeated) offspring generated
    unique_offspring1 = unique_offspring1 or offspring1
    unique_offspring2 = unique_offspring2 or offspring2

    return unique_offspring1, unique_offspring2
