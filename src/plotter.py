import pygraphviz as pgv
import matplotlib.pyplot as plt
from deap import gp
import numpy as np
import os
import itertools

def plot_model(model, file_name):

    # Make the required directories if they dont exist
    _make_directories(file_name)

    nodes, edges, labels = gp.graph(model)

    # For simpler graphs we want to ignore masks as they only used for the code
    to_remove = [idx for idx in labels if labels[idx] == "Mask"]

    # We also want to simplify the tree where we can
    for idx in labels:
        label = str(labels[idx])

        # Skip as we iterate.
        if idx in to_remove:
            continue

        # If its a feature node, do some pretty formatting and add the split point to the node
        if label.startswith("FN_"):
            # The split value is always the next child
            split_child = idx + 1
            to_remove.append(split_child)  # The child no longer needs to be its own node

            split_value = round(labels[split_child], 2)
            clean_label = label.replace("FN_", "").replace("Feature", "Feature ").replace("gt", " > ")

            labels[idx] = clean_label + str(split_value)
        elif label.endswith("_le") or label.endswith("_gt"):
            # We should remove these nodes by replacing them with their only child
            # These are only used for the program but not important visually
            to_remove.append(idx)

            # They only have one child
            replacement = idx + 1

            # Update the edges
            edges = [(replacement, edge[1]) if edge[0] == idx else edge for edge in edges]
            edges = [(edge[0], replacement) if edge[1] == idx else edge for edge in edges]

            # We would have introduced a self edge doing this
            edges = [edge for edge in edges if edge[0] != edge[1]]
        elif label == "ConstructedFilter":

            # The first child is the constructed feature
            constructed_child = idx + 1

            label, remove = _flatten_constructed(constructed_child, nodes, edges, labels)
            label = label + " >= 0"

            labels[idx] = label

            to_remove = to_remove + remove

    # Remove the redundant nodes
    nodes = [node for node in nodes if node not in to_remove]
    edges = [edge for edge in edges if edge[0] not in to_remove and edge[1] not in to_remove]

    g = pgv.AGraph(outputorder="edgesfirst")
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    # Format the graph
    for i in nodes:
        n = g.get_node(i)
        label = str(labels[i])

        n.attr["label"] = label
        n.attr["fillcolor"] = "white"
        n.attr["style"] = "filled"

    g.draw(file_name)


def plot_pareto(frontier, file_name):
    _make_directories(file_name)

    frontier = np.array([ind.fitness.values for ind in frontier])
    plt.scatter(frontier[:, 1], frontier[:, 0], c="b")
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