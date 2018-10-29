from helpers import read_data
from src.forest import EvolutionaryForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

kf = StratifiedKFold(shuffle=True, n_splits=2, random_state=0)
fold = 0

X, y = read_data("data/iris.data")

train_index, test_index = list(kf.split(X, y))[fold]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average="weighted")
    print(model.__class__.__name__, score)
    return score


evoTree = EvolutionaryForest(max_trees=5, num_generations=5)
evoTree.fit(X_train, y_train)
ensemble_preds = evoTree.predict_majority(X_test)

preds = evoTree.predict(X_test)

print(ensemble_preds == preds)

print(X_test.shape)

print(ensemble_preds)
print(preds)

'''
dt = DecisionTreeClassifier()



proposed = evaluate(evoTree, X_train, X_test, y_train, y_test)
comparison = evaluate(dt, X_train, X_test, y_train, y_test)

print("%.3f" % proposed, "%.3f" % comparison)
'''