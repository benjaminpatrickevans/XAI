from helpers import read_data, CategoricalToNumeric
from src.xai import GP
from comparisons import InterpretableDecisionTreeClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
import sys
import subprocess
import pickle
import os
import pandas as pd
import pysbrl
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

from h2o.backend import H2OLocalServer
import h2o
h2o.init()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def scorer(predicted, real):
    return f1_score(real, predicted, average="weighted")


def sklearn_plot(model, model_file_prefix):
    export_graphviz(model, out_file=model_file_prefix + "_sdt.gv", impurity=False, label="root", precision=2,
                    class_names=model.classes_)

    png_file_name = model_file_prefix + '_sdt.png'
    png_args = str('dot -Tpng ' + model_file_prefix + "_sdt.gv" + ' -o ').split()
    png_args.append(png_file_name)

    subprocess.call(png_args)


def h2o_plot(model, model_file_prefix):
    '''
        Plot an h2o tree. From:
        https://resources.oreilly.com/oriole/interpretable-machine-learning-with-python-xgboost-and-h2o/blob/master/dt_surrogate_loco.ipynb

    :param model:
    :param model_file_prefix:
    :return:
    '''
    mojo_path = model.download_mojo()

    hs = H2OLocalServer()
    h2o_jar_path = hs._find_jar()

    gv_file_name = model_file_prefix + '.gv'
    gv_args = str('java -cp ' + h2o_jar_path +
                  ' hex.genmodel.tools.PrintMojo --tree 0 --decimalplaces 2 -i '
                  + mojo_path + ' -o').split()
    gv_args.append(gv_file_name)

    subprocess.call(gv_args)

    # Compute complexity of dt based on the resulting graphviz
    complexity = open(gv_file_name, 'r').read().count("shape=box")

    png_file_name = model_file_prefix + '.png'
    png_args = str('dot -Tpng ' + gv_file_name + ' -o ').split()
    png_args.append(png_file_name)

    subprocess.call(png_args)

    return complexity


def decision_tree(X_train, y_train, X_test, y_test):
    h2_blackbox_train = h2o.H2OFrame(python_obj=np.hstack((X_train, y_train)))
    h2_blackbox_test = h2o.H2OFrame(python_obj=np.hstack((X_test, y_test)))

    model_id = 'dt_surrogate_mojo'
    dt = H2ORandomForestEstimator(ntrees=1, sample_rate=1, mtries=-2, model_id=model_id,
                                  max_depth=6)  # Make a h2o decision tree. Same max depth as our models

    # Train using the predictions from the RF
    dt.train(x=h2_blackbox_train.columns[:-1], y=h2_blackbox_train.columns[-1], training_frame=h2_blackbox_train)

    training_recreations = dt.predict(h2_blackbox_train)["predict"].as_data_frame().values
    dt_training_recreating_pct = scorer(training_recreations, y_train) * 100
    testing_recreations = dt.predict(h2_blackbox_test)["predict"].as_data_frame().values
    dt_testing_recreating_pct = scorer(testing_recreations, y_test) * 100

    dt_complexity = h2o_plot(dt, model_file + "-%.2f_dt" % dt_testing_recreating_pct)

    return dt_training_recreating_pct, dt_testing_recreating_pct, dt_complexity

def decision_stump(X_train, y_train, X_test, y_test):
    h2_blackbox_train = h2o.H2OFrame(python_obj=np.hstack((X_train, y_train)))
    h2_blackbox_test = h2o.H2OFrame(python_obj=np.hstack((X_test, y_test)))

    model_id = 'dt_stump_mojo'
    dt = H2ORandomForestEstimator(ntrees=1, sample_rate=1, mtries=-2, model_id=model_id,
                                  max_depth=1)  # Make a h2o decision stump

    # Train using the predictions from the RF
    dt.train(x=h2_blackbox_train.columns[:-1], y=h2_blackbox_train.columns[-1], training_frame=h2_blackbox_train)

    training_recreations = dt.predict(h2_blackbox_train)["predict"].as_data_frame().values
    dt_training_recreating_pct = scorer(training_recreations, y_train) * 100
    testing_recreations = dt.predict(h2_blackbox_test)["predict"].as_data_frame().values
    dt_testing_recreating_pct = scorer(testing_recreations, y_test) * 100

    dt_complexity = h2o_plot(dt, model_file + "-%.2f_ds" % dt_testing_recreating_pct)

    return dt_training_recreating_pct, dt_testing_recreating_pct, dt_complexity


def apply_one_hot_encoding(X_train, X_test):
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    encoder = CategoricalToNumeric()

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train.values, X_test.values

def simplified_decision_tree(X_train, y_train, X_test, y_test):

    X_train, X_test = apply_one_hot_encoding(X_train, X_test)

    # Simplified Sklearn decision tree
    sdt = InterpretableDecisionTreeClassifier.IDecisionTreeClassifier()
    sdt.fit(X_train, y_train)

    sdt_complexity = str(sdt).count("if")  # Number of if statements = number of splitting points

    training_recreations = sdt.predict(X_train)
    sdt_training_recreating_pct = scorer(training_recreations, y_train) * 100
    testing_recreations = sdt.predict(X_test)
    sdt_testing_recreating_pct = scorer(testing_recreations, y_test) * 100

    sklearn_plot(sdt, model_file + "-%.2f" % sdt_testing_recreating_pct)

    return sdt_training_recreating_pct, sdt_testing_recreating_pct, sdt_complexity


def logistic_regression(X_train, y_train, X_test, y_test):
    X_train, X_test = apply_one_hot_encoding(X_train, X_test)

    # Logistic Regression
    lr = LogisticRegression(penalty='l1')
    lr.fit(X_train, y_train)
    lr_complexity = np.count_nonzero(lr.coef_)

    print("Logistic", lr.intercept_, lr.coef_)

    training_recreations = lr.predict(X_train)
    lr_training_recreating_pct = scorer(training_recreations, y_train) * 100
    testing_recreations = lr.predict(X_test)
    lr_testing_recreating_pct = scorer(testing_recreations, y_test) * 100

    return lr_training_recreating_pct, lr_testing_recreating_pct, lr_complexity


def genetic_program(X_train, y_train, X_test, y_test, num_generations,
                    num_trees, model_file):

    evoTree = GP(max_trees=num_trees, num_generations=num_generations)
    evoTree.fit(X_train, y_train)
    gp_complexity = evoTree.complexity()

    training_recreations = evoTree.predict(X_train)
    gp_training_recreating_pct = scorer(training_recreations, y_train) * 100
    testing_recreations = evoTree.predict(X_test)
    gp_testing_recreating_pct = scorer(testing_recreations, y_test) * 100

    evoTree.plot(model_file + "-%.2f.png" % gp_testing_recreating_pct) # Save the resulting tree
    evoTree.plot_pareto(model_file + "_pareto.png")

    return gp_training_recreating_pct, gp_testing_recreating_pct, gp_complexity


def bayesian_rule_list(X_train, y_train, X_test, y_test):
    from mdlp.discretization import MDLP
    from sklearn import preprocessing

    # First one hot encode
    X_train, X_test = apply_one_hot_encoding(X_train, X_test)


    # Then need to convert classes to integers
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Then discretize features
    transformer = MDLP()
    X_train = transformer.fit_transform(X_train, y_train)
    X_test = transformer.transform(X_test)

    brl = pysbrl.BayesianRuleList()
    brl.fit(X_train, y_train)

    print(brl)

    # The complexity is the number of split points + the number of extra conditions
    # (i.e. if x1 > 0 and x2 = 1 then .. counts as 2 not 1), for this reason we do not use brl.n_rules
    brl_str = str(brl)
    brl_complexity = brl_str.count("IF") + brl_str.count("AND")

    training_recreations = brl.predict(X_train)
    brl_training_recreating_pct = scorer(training_recreations, y_train) * 100
    testing_recreations = brl.predict(X_test)
    brl_testing_recreating_pct = scorer(testing_recreations, y_test) * 100

    return brl_training_recreating_pct, brl_testing_recreating_pct, brl_complexity


def main(data, num_generations, num_trees, fold, seed, model_file, blackbox_model):
    ###########
    kf = StratifiedKFold(shuffle=True, n_splits=10, random_state=seed)
    X, y = read_data("data/"+data+".csv")

    # Split the data based on the fold of this run
    train_index, test_index = list(kf.split(X, y))[fold]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # H2o requires specially formatted data
    h2_train = h2o.H2OFrame(python_obj=np.hstack((X_train, y_train)))
    h2_test = h2o.H2OFrame(python_obj=np.hstack((X_test, y_test)))

    # =================
    # Train the complex model
    # =================

    blackbox_options = {
        "RF": H2ORandomForestEstimator(ntrees=100),
        "GB": H2OGradientBoostingEstimator(ntrees=100),
        "DL": H2ODeepLearningEstimator(epochs=1000),
    }

    # Choose the model based on the given parameter
    blackbox = blackbox_options[blackbox_model]

    blackbox.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)

    # We use the predictions from the model as the new "labels" for training surrogate.
    blackbox_train_predictions = blackbox.predict(h2_train)["predict"].as_data_frame().values
    blackbox_train_score = scorer(blackbox_train_predictions, y_train)

    blackbox_test_predictions = blackbox.predict(h2_test)["predict"].as_data_frame().values
    blackbox_test_score = scorer(blackbox_test_predictions, y_test)

    print("The " + blackbox.__class__.__name__ + " achieved", "%.2f" % blackbox_train_score, "on the train set and",
          "%.2f" % blackbox_test_score, "on the test set")

    # =================
    # Train the surrogates
    # =================

    dt_training_recreating_pct, dt_testing_recreating_pct, dt_complexity = \
        decision_tree(X_train, blackbox_train_predictions, X_test, blackbox_test_predictions)

    print("DT was able to recreate %.2f%%" % dt_training_recreating_pct, "of them on the train, and %.2f%%" %
          dt_testing_recreating_pct, "on the test set")

    sdt_training_recreating_pct, sdt_testing_recreating_pct, sdt_complexity = \
        simplified_decision_tree(X_train, blackbox_train_predictions, X_test, blackbox_test_predictions)

    print("SDT was able to recreate %.2f%%" % sdt_training_recreating_pct, "of them on the train, and %.2f%%" %
          sdt_testing_recreating_pct, "on the test set")

    ds_training_recreating_pct, ds_testing_recreating_pct, ds_complexity = \
        decision_stump(X_train, blackbox_train_predictions, X_test, blackbox_test_predictions)

    print("DS was able to recreate %.2f%%" % ds_training_recreating_pct, "of them on the train, and %.2f%%" %
          ds_testing_recreating_pct, "on the test set")

    lr_training_recreating_pct, lr_testing_recreating_pct, lr_complexity = \
        logistic_regression(X_train, blackbox_train_predictions, X_test, blackbox_test_predictions)

    print("LR was able to recreate %.2f%%" % lr_training_recreating_pct, "of them on the train, and %.2f%%" %
          lr_testing_recreating_pct, "on the test set")

    '''
    brl_training_recreating_pct, brl_testing_recreating_pct, brl_complexity = \
        bayesian_rule_list(X_train, blackbox_train_predictions, X_test, blackbox_test_predictions)

    print("BRL was able to recreate %.2f%%" % brl_training_recreating_pct, "of them on the train, and %.2f%%" %
          brl_testing_recreating_pct, "on the test set")
          '''

    gp_training_recreating_pct, gp_testing_recreating_pct, gp_complexity = \
        genetic_program(X_train, blackbox_train_predictions, X_test, blackbox_test_predictions, num_generations,
                        num_trees, model_file)

    print("GP was able to recreate %.2f%%" % gp_training_recreating_pct, "of them on the train, and %.2f%%" %
          gp_testing_recreating_pct, "on the test set")

    return [blackbox_train_score, blackbox_test_score,
            dt_training_recreating_pct, dt_testing_recreating_pct, dt_complexity,
            sdt_training_recreating_pct, sdt_testing_recreating_pct, sdt_complexity,
            ds_training_recreating_pct, ds_testing_recreating_pct, ds_complexity,
            lr_training_recreating_pct, lr_testing_recreating_pct, lr_complexity,
            brl_training_recreating_pct, brl_testing_recreating_pct, brl_complexity,
            gp_training_recreating_pct, gp_testing_recreating_pct, gp_complexity]


def save_results_to_file(res, out_file):
    # Save
    with open(out_file, 'wb') as fp:
        pickle.dump(res, fp)


if __name__ == "__main__":
    print(sys.argv)
    data = sys.argv[1]
    num_generations = int(sys.argv[2])
    num_trees = int(sys.argv[3])
    fold = int(sys.argv[4])
    seed = int(sys.argv[5])
    blackbox_model = sys.argv[6].upper()

    out_dir = "out/%s/" % data
    res_file = out_dir + "results-multi-%s-g%d-t%d-f%d-s%d.pickle" % (blackbox_model, num_generations, num_trees, fold, seed)
    model_file = out_dir + "model-multi-%s-g%d-t%d-f%d-s%d" % (blackbox_model, num_generations, num_trees, fold, seed)

    if False and os.path.exists(res_file):  # TODO: Uncomment when running properly
        print("Have already ran for these settings, exiting early")
    else:

        # Make the subdirectory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run and save results
        res = main(data, num_generations, num_trees, fold, seed, model_file, blackbox_model)
        print(res)
        save_results_to_file(res, res_file)


