from helpers import read_data
from src.forest import EvolutionaryForest
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from h2o.estimators.random_forest import H2ORandomForestEstimator
import h2o
h2o.init()

###########
num_trees = 10

###########
kf = StratifiedKFold(shuffle=True, n_splits=2, random_state=0)
fold = 0

X, y = read_data("data/iris.data")

train_index, test_index = list(kf.split(X, y))[fold]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

num_features = X_train.shape[1]
full_train = np.hstack((X_train, y_train))
full_test = np.hstack((X_test, y_test))

# H2o requires specially formatted data
h2_train = h2o.H2OFrame(python_obj=full_train)
h2_test = h2o.H2OFrame(python_obj=full_test)

###########

def evaluate(name, preds):
    score = f1_score(y_test, preds, average="weighted")
    print(name, score)
    return score

##########

dt = H2ORandomForestEstimator(ntrees=1, sample_rate =1, mtries=num_features) # Setup RF like a decision tree
dt.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)
dt_preds = dt.predict(h2_test)["predict"].as_data_frame().values
evaluate("Decision Tree", dt_preds)


rf = H2ORandomForestEstimator(ntrees=num_trees)
rf.train(x=h2_train.columns[:-1], y=h2_train.columns[-1], training_frame=h2_train)
rf_preds = rf.predict(h2_test)["predict"].as_data_frame().values
evaluate("Random Forest", rf_preds)


evoTree = EvolutionaryForest(max_trees=num_trees, num_generations=100)
evoTree.fit(X_train, y_train)
preds = evoTree.predict(X_test)
ensemble_preds = evoTree.predict_majority(X_test)
weighted_ensemble_preds = evoTree.predict_weighted_majority(X_test)
evaluate("Evolutionary Tree", preds)
evaluate("Evolutionary Forest (Majority)", ensemble_preds)
evaluate("Evolutionary Forest (Weighted Majority)", weighted_ensemble_preds)