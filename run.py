from helpers import read_data
from src.forest import EvolutionaryForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

###########

kf = StratifiedKFold(shuffle=True, n_splits=2, random_state=0)
fold = 0

X, y = read_data("data/iris.data")

train_index, test_index = list(kf.split(X, y))[fold]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

def evaluate(name, preds):
    score = f1_score(y_test, preds, average="weighted")
    print(name, score)
    return score

##########

evoTree = EvolutionaryForest(max_trees=50, num_generations=10)
evoTree.fit(X_train, y_train)
preds = evoTree.predict(X_test)
ensemble_preds = evoTree.predict_majority(X_test)
evaluate("Evolutionary Tree", preds)
evaluate("Evolutionary Forest (Majority)", ensemble_preds)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
evaluate("Decision Tree", dt_preds)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
evaluate("Random Forest", rf_preds)
