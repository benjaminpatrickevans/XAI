import pygraphviz as pgv
import matplotlib
matplotlib.use('Agg') # Do not use X windows since running as a script
import matplotlib.pyplot as plt
from deap import gp
import numpy as np
import os
import re
import operator
from functools import partial, reduce

def _update_edges(edges, replacement, idx):
    edges = [(replacement, edge[1]) if edge[0] == idx else edge for edge in edges]
    edges = [(edge[0], replacement) if edge[1] == idx else edge for edge in edges]

    # We would have introduced a self edge doing this
    edges = [edge for edge in edges if edge[0] != edge[1]]

    return edges

def _path_to_root(idx, edges, labels):
    parents = [parent for (parent, child) in edges if child == idx]

    if len(parents) > 1:
        print("There should not be more than 2 parents. Something is wrong")
        return None

    if not parents:
        # Must be at the root
        return []

    parent = parents[0]

    path = [parent]

    if labels[parent].startswith("FN_NumericFeature"):
        # Add the splitting point as well
        path.append(parent + 1)

    # Recurse on parent
    return path + _path_to_root(parent, edges, labels)


def _most_common_class(data):
    # Using the matching data, if there was none then just use the training set
    labels = data[:, -1]

    # Count the occurences for each class
    classes, class_counts = np.unique(labels, return_counts=True)

    # Find the class that achieved the most counts
    most_common_class_idx = np.argmax(class_counts)

    # Return this class
    return classes[most_common_class_idx]


def _matches_condition(op, feature_index, data):
    # Return a boolean array where true indicates op was met, false indicates op was not met
    return np.where(op(data[:, feature_index]))


def _path_to_functions(path, nodes, edges, labels):

    operations = []

    for idx, node in enumerate(path):
        label = str(labels[node])

        if label.endswith("_le") or label.endswith("_gt"):
            # Numeric feature splits
            feature = str(labels[path[idx + 1]])
            feature_idx = int(feature.split("FN_NumericFeature")[1])  # Just extract the index
            splitting_point = float(labels[path[idx + 2]])
            op = operator.le if label.endswith("_le") else operator.gt

            operations.append(partial(_matches_condition, partial(op, splitting_point), feature_idx))
        elif "_category" in label:
            # Categorical feature splits
            # TODO: This will break if the "clean" category name used for the node differs to the original category name
            # for example if original name had (.) or (+) then we removed these for the tree.
            category = label.split("_category")[1]
            feature = str(labels[path[idx + 1]])
            feature_idx = int(feature.split("FN_CategoryFeature")[1])  # Just extract the index

            # Since binary features are treated as categorical
            if category.isdigit():
                category = int(category)

            operations.append(partial(_matches_condition, partial(operator.eq, category), feature_idx))

        elif label == "ConstructedFilter":
            # Constructed filter splits
            constructed_child = node + 1
            flat_label, _ = _flatten_constructed(constructed_child, nodes, edges, labels)

            # We want to turn a flattened label, i.e. (CFN_Feature1*CFN_Feature3) into a callable function
            # replace CFN_FeatureX with data[:, x] (i.e. access the columns in a nd array).
            flat_label = re.sub(r'(CFN_Feature)([0-9]+)', r'data[:, \2]', flat_label)

            # Make a function which takes in the data, and will apply the function string from above
            fn = eval("lambda data: np.where(" + flat_label + "> 0)")
            operations.append(fn)

    return operations


def plot_model(model, file_name, train_data):

    # Make the required directories if they dont exist
    _make_directories(file_name)

    nodes, edges, labels = gp.graph(model)

    # For simpler graphs we want to ignore masks as they only used for the code
    to_remove = [idx for idx in labels if labels[idx] == "Mask"]

    edge_labels = {}

    leaves = [idx for idx in labels if labels[idx] == "TrainData"]

    # Update the leaves to have the class labels
    for leaf in leaves:

        # Find the path to the root, and apply all the conditions along the way
        path = _path_to_root(leaf, edges, labels)

        conditions = _path_to_functions(path, nodes, edges, labels)
        filtered_indices = [condition(train_data) for condition in conditions]

        if filtered_indices:

            if len(filtered_indices) > 1:
                # Need to find the data that matches ALL conditions, i.e. the intersect of all filters above
                matching_indices = reduce(np.intersect1d, filtered_indices)
            else:
                # Special case when theres only one, a tuple is returned so we need the first element of the list, then
                # the first element of this tuple.
                matching_indices = filtered_indices[0][0]

            # Apply this to the training data. If theres no matching data, use majority class overall
            matching_data = train_data[matching_indices] if matching_indices.size else train_data
            leaf_class = _most_common_class(matching_data)

            # Set the label to be the class
            labels[leaf] = leaf_class


    # We also want to simplify the tree where we can
    for idx in labels:
        label = str(labels[idx])

        # Skip as we iterate.
        if idx in to_remove:
            continue

        # If its a feature node, do some pretty formatting and add the split point to the node
        if label.startswith("FN_Numeric"):
            # The split value is always the next child
            split_child = idx + 1
            to_remove.append(split_child)  # The child no longer needs to be its own node
            split_value = round(labels[split_child], 2)
            label = label.replace("FN_NumericFeature", "C")
            label = label + " > " + str(split_value)
            labels[idx] = label
        elif label.endswith("_le") or label.endswith("_gt"):
            # We should remove these nodes by replacing them with their only child
            # These are only used for the program but not important visually
            to_remove.append(idx)

            # They only have one child
            replacement = idx + 1

            # Replace all idx edges with the replacement node
            edges = _update_edges(edges, replacement, idx)

        elif label == "ConstructedFilter":

            # The first child is the constructed feature
            constructed_child = idx + 1

            label, remove = _flatten_constructed(constructed_child, nodes, edges, labels)
            label = label + " > 0"

            labels[idx] = label

            to_remove = to_remove + remove

        elif "_category" in label:
            # We want to replace categorical nodes with just an edge saying the category
            to_remove.append(idx)

            # Only one child (the category)
            replacement = idx + 1

            # Replace all idx edges with the replacement node
            edges = _update_edges(edges, replacement, idx)

            incoming_edge = next(edge for edge in edges if edge[1] == replacement)

            # We just want the category name, not any of the prefix
            label = label.split("_category")[1]

            edge_labels[incoming_edge] = label


    # Now we need to update the constructed leaves to have class labels. Couldnt do this at the start as we want
    # the flattened labels

    # Remove the redundant nodes
    nodes = [node for node in nodes if node not in to_remove]
    edges = [edge for edge in edges if edge[0] not in to_remove and edge[1] not in to_remove]

    g = pgv.AGraph(outputorder="edgesfirst", directed=True, nodesep=.5, ranksep=1)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    # Format the graph
    for i in nodes:
        n = g.get_node(i)
        label = str(labels[i])

        # Try and match the type outputted by h2o for fair comparison

        # If its a numeric splitting point
        if ">" in label:
            # Match the type outputted by h2o for fair comparison
            label = label.replace("CFN_Feature", "C")

            n.attr["shape"] = "box"

            edges = g.edges(n)
            edges[0].attr["label"] = "\n<=\n"
            edges[0].attr["fontsize"] = 14
            edges[1].attr["label"] = "\n      >\n"
            edges[1].attr["fontsize"] = 14

        elif label.startswith("FN_Category"):
            label = label.replace("FN_CategoryFeature", "C")
            n.attr["shape"] = "box"

            # Add the categories as edge labels
            edges = g.edges(n)

            for edge in edges:
                edge_indices = int(edge[0]), int(edge[1])
                # Need to convert to ints since pygraphviz uses strings
                if edge_indices in edge_labels:
                    edge.attr["label"] = edge_labels[edge_indices]


        n.attr["label"] = label
        n.attr["fillcolor"] = "white"
        n.attr["fontsize"] = 14
        n.attr["style"] = "filled"

    g.draw(file_name)


def plot_pareto(frontier, population, file_name):
    _make_directories(file_name)

    frontier = np.array([ind.fitness.values for ind in frontier])
    population = np.array([ind.fitness.values for ind in population])

    plt.scatter(population[:, 1], population[:, 0], c="b")
    plt.scatter(frontier[:, 1], frontier[:, 0], c="r")
    plt.ylabel("f1-score")
    plt.xlabel("Size")

    # Both in range 0..1
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.savefig(file_name)


def _flatten_constructed(idx, nodes, edges, labels):
    """
        Used for plotting, rather than a tree having a subtree
        for a constructed feature, this gets flattened to a single
        node.
    :param idx:
    :param nodes:
    :param edges:
    :param labels:
    :return:
    """
    children = [child for (parent, child) in edges if parent == idx]

    if not children:
        return labels[idx], [idx]

    replacements = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/"
    }

    label = labels[idx]
    label = label.replace(label, replacements[label])

    children_out = [_flatten_constructed(child, nodes, edges, labels) for child in children]

    children_labels = [out[0] for out in children_out]
    children_indices = [out[1] for out in children_out]

    indices_to_remove = [idx]

    # Add all the children index lists to our list
    for children_idx_list in children_indices:
        indices_to_remove.extend(children_idx_list)

    # Return an updated label and the children nodes so we can remove them
    return "(" + children_labels[0] + label + children_labels[1] + ")", indices_to_remove


def _make_directories(file_name):
    file_dir = "/".join(file_name.split("/")[:-1])  # Extract the directory from the file name

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)