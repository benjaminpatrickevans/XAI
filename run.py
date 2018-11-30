from helpers import read_data
from src.xai import GP
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import sys
import subprocess
import pickle
import os
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

from h2o.backend import H2OLocalServer
import h2o
h2o.init()


def h2o_plot(model, model_file):
    '''
        Plot an h2o tree. From:
        https://resources.oreilly.com/oriole/interpretable-machine-learning-with-python-xgboost-and-h2o/blob/master/dt_surrogate_loco.ipynb

    :param model:
    :param model_file:
    :return:
    '''
    mojo_path = model.download_mojo(path='.')

    hs = H2OLocalServer()
    h2o_jar_path = hs._find_jar()

    gv_file_name = model_file + '_dt.gv'
    gv_args = str('java -cp ' + h2o_jar_path +
                  ' hex.genmodel.tools.PrintMojo --tree 0 -i '
                  + mojo_path + ' -o').split()
    gv_args.append(gv_file_name)

    subprocess.call(gv_args)

    png_file_name = model_file + '_dt.png'
    png_args = str('dot -Tpng ' + gv_file_name + ' -o ').split()
    png_args.append(png_file_name)

    subprocess.call(png_args)


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
    blackbox_train_score = f1_score(blackbox_train_predictions, y_train, average="weighted")

    blackbox_test_predictions = blackbox.predict(h2_test)["predict"].as_data_frame().values
    blackbox_test_score = f1_score(blackbox_test_predictions, y_test, average="weighted")

    print("The " + blackbox.__class__.__name__ + " achieved", "%.2f" % blackbox_train_score, "on the train set and",
          "%.2f" % blackbox_test_score, "on the test set")

    # =================
    # Train the surrogates
    # =================

    h2_blackbox_train = h2o.H2OFrame(python_obj=np.hstack((X_train, blackbox_train_predictions)))
    h2_blackbox_test = h2o.H2OFrame(python_obj=np.hstack((X_test, blackbox_test_predictions)))

    model_id = 'dt_surrogate_mojo'
    dt = H2ORandomForestEstimator(ntrees=1, sample_rate=1, mtries=-2, model_id=model_id,
                                  max_depth=6)  # Make a h2o decision tree. Same max depth as our models

    # Train using the predictions from the RF
    dt.train(x=h2_blackbox_train.columns[:-1], y=h2_blackbox_train.columns[-1], training_frame=h2_blackbox_train)
    h2o_plot(dt, model_file)

    training_recreations = dt.predict(h2_blackbox_train)["predict"].as_data_frame().values
    dt_training_recreating_pct = accuracy_score(training_recreations, blackbox_train_predictions) * 100
    testing_recreations = dt.predict(h2_blackbox_test)["predict"].as_data_frame().values
    dt_testing_recreating_pct = accuracy_score(testing_recreations, blackbox_test_predictions) * 100

    print("DT was able to recreate %.2f%%" % dt_training_recreating_pct, "of them on the train, and %.2f%%" %
          dt_testing_recreating_pct, "on the test set")

    # Proposed
    evoTree = GP(max_trees=num_trees, num_generations=num_generations)
    evoTree.fit(X_train, blackbox_train_predictions)
    evoTree.plot(model_file+".png") # Save the resulting tree
    evoTree.plot_pareto(model_file+"_pareto.png")
    training_recreations = evoTree.predict(X_train)
    gp_training_recreating_pct = accuracy_score(training_recreations, blackbox_train_predictions) * 100

    testing_recreations = evoTree.predict(X_test)
    gp_testing_recreating_pct = accuracy_score(testing_recreations, blackbox_test_predictions) * 100

    print("GP was able to recreate %.2f%%" % gp_training_recreating_pct, "of them on the train, and %.2f%%" %
          gp_testing_recreating_pct, "on the test set")

    return [blackbox_train_score, blackbox_test_score, dt_training_recreating_pct, dt_testing_recreating_pct,
            gp_training_recreating_pct, gp_testing_recreating_pct]


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
