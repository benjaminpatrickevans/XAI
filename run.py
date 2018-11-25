from helpers import read_data
from src.xai import GP
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import sys
import pickle
import os
from h2o.estimators.random_forest import H2ORandomForestEstimator
import h2o
h2o.init()

def xgboost():
    pass

def rf():
    pass

def deep_learning():
    pass

def main(data, num_generations, num_trees, fold, seed, model_file):
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

    # Make and train the RF
    model = H2ORandomForestEstimator(ntrees=100)
    model.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)

    # We use the predictions from the model as the new "labels" for training GP. Important to not touch test
    # yet to avoid any bias
    blackbox_train_predictions = model.predict(h2_train)["predict"].as_data_frame().values
    blackbox_train_score = f1_score(blackbox_train_predictions, y_train, average="weighted")

    evoTree = GP(max_trees=num_trees, num_generations=num_generations)
    evoTree.fit(X_train, blackbox_train_predictions)
    training_recreations = evoTree.predict(X_train)
    training_recreating_pct = accuracy_score(training_recreations, blackbox_train_predictions) * 100

    print("The random forest achieved", "%.2f" % blackbox_train_score, "on the train set")
    print("Now training GP on these predictions")
    print("GP was able to recreate", training_recreating_pct, "% of them")

    blackbox_test_predictions = model.predict(h2_test)["predict"].as_data_frame().values
    blackbox_test_score = f1_score(blackbox_test_predictions, y_test, average="weighted")
    testing_recreations = evoTree.predict(X_test)
    testing_recreating_pct = accuracy_score(testing_recreations, blackbox_test_predictions) * 100

    print("On the test set, the RF achieved %.2f" % blackbox_test_score)
    print("And the new model was able to recreate", testing_recreating_pct, "% of them")
    evoTree.plot(model_file)

    return [blackbox_train_score, training_recreating_pct, blackbox_test_score, testing_recreating_pct]


def save_results_to_file(res, out_dir, out_file):
    # Make the subdirectory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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

    out_dir = "out/%s/" % data
    res_file = out_dir + "results-multi-g%d-t%d-f%d-s%d.pickle" % (num_generations, num_trees, fold, seed)
    model_file = out_dir + "model-multi-g%d-t%d-f%d-s%d.png" % (num_generations, num_trees, fold, seed)

    if False and os.path.exists(res_file):  # TODO: Uncomment when running properly
        print("Have already ran for these settings, exiting early")
    else:
        # Run and save results
        res = main(data, num_generations, num_trees, fold, seed, model_file)
        print(res)
        save_results_to_file(res, out_dir, res_file)
