from helpers import read_data, evaluate
from src.forest import EvolutionaryForest
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
import pickle
import os
from h2o.estimators.random_forest import H2ORandomForestEstimator
import h2o
h2o.init()

'''
breast-cancer first
congress first
contraceptive last
dresses last
german last
haberman last
planning last
post-operative last
primary-tumor first
spect first
statlog-aus last
swiss last
thoracic last
titanic last * changed for this, used to be first
va last
wpbc last
iris last
'''


def main(data, num_generations, num_trees, fold, seed):
    ###########
    kf = StratifiedKFold(shuffle=True, n_splits=10, random_state=seed)
    X, y = read_data("data/"+data+".csv")

    train_index, test_index = list(kf.split(X, y))[fold]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    num_features = X_train.shape[1]
    full_train = np.hstack((X_train, y_train))
    full_test = np.hstack((X_test, y_test))

    # H2o requires specially formatted data
    h2_train = h2o.H2OFrame(python_obj=full_train)
    h2_test = h2o.H2OFrame(python_obj=full_test)

    ##########
    dt = H2ORandomForestEstimator(ntrees=1, sample_rate =1, mtries=num_features) # Setup RF like a decision tree
    dt.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)
    dt_preds = dt.predict(h2_test)["predict"].as_data_frame().values
    dt_score = evaluate("Decision Tree", dt_preds, y_test)

    rf = H2ORandomForestEstimator(ntrees=num_trees)
    rf.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)
    rf_preds = rf.predict(h2_test)["predict"].as_data_frame().values
    rf_score = evaluate("Random Forest", rf_preds, y_test)

    xt = H2ORandomForestEstimator(ntrees=num_trees, histogram_type="Random")
    xt.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)
    xt_preds = xt.predict(h2_test)["predict"].as_data_frame().values
    xt_score = evaluate("Extremely Randomized", xt_preds, y_test)

    evoTree = EvolutionaryForest(max_trees=num_trees, num_generations=num_generations)
    evoTree.fit(X_train, y_train)
    preds = evoTree.predict(X_test)
    ensemble_preds = evoTree.predict_majority(X_test)
    weighted_ensemble_preds = evoTree.predict_weighted_majority(X_test)
    evo_score = evaluate("Evolutionary Tree", preds, y_test)
    evo_forest_score = evaluate("Evolutionary Forest (Majority)", ensemble_preds, y_test)
    evo_weighted_forest_score = evaluate("Evolutionary Forest (Weighted Majority)", weighted_ensemble_preds, y_test)

    results = [dt_score, rf_score, xt_score, evo_score, evo_forest_score, evo_weighted_forest_score]
    print("Results", ['%.3f' % elem for elem in results])
    return results


def save_to_file(res, out_dir, out_file):
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
    out_file = out_dir + "results-multi-g%d-t%d-f%d-s%d.pickle" % (num_generations, num_trees, fold, seed)

    if False and os.path.exists(out_file):  # TODO: Uncomment when running properly
        print("Have already ran for these settings, exiting early")
    else:
        # Run and save results
        res = main(data, num_generations, num_trees, fold, seed)
        save_to_file(res, out_dir, out_file)
