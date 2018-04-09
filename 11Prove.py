from sklearn import ensemble
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import math


def load_lens():
    # Load Lens Data
    headers = ["age", "prescription", "astigmatic", "tear_rate", "lens"]
    lenses = pd.read_csv("lenses.data.txt", header=None, names=headers, na_values="?", delim_whitespace=True,
                         index_col=0)
    lenses.head()

    lens_target = lenses["lens"]
    lens_data = lenses.drop("lens", axis=1)

    print("Lens Data")
    test(lens_data, lens_target)


def load_iris():
    iris = datasets.load_iris()
    print("Iris Data")
    test(iris.data, iris.target)


def load_abalone():
    headers = ["sex", "length", "diameter", "height", "whole_height", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
    abalone = pd.read_csv("abalone.data.txt", header=None, names=headers, na_values="?")

    cleanup = {"sex": {"M": 0, "F": 1, "I": 2}}
    abalone.replace(cleanup, inplace=True)
    abalone.head()
    print(abalone.values[0:4])

    abalone_target = abalone["rings"]
    abalone_data = abalone.drop("rings", axis=1)
    abalone_data = normalize(abalone_data, axis=0, copy=False)
    print(abalone_data[0:4])

    print("Abalone Data")
    test(abalone_data, abalone_target)


def test(data, targets):
    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.3)

    # Use sk-learn tree classifier
    results_tree = predict_tree(data_train, data_test, targets_train)

    # KNN
    results_knn = predict_knn(data_train, data_test, targets_train)

    # MLP
    results_mlp = predict_mlp(data_train, data_test, targets_train)

    # Bagging
    bag_tree, bag_knn, bag_mlp = bagging(data_train, data_test, targets_train)

    # Results
    print("Tree:     " + results(targets_test, results_tree))
    print("KNN:      " + results(targets_test, results_knn))
    print("MLP:      " + results(targets_test, results_mlp))
    print("Bag Tree: " + results(targets_test, bag_tree))
    print("Bag KNN:  " + results(targets_test, bag_knn))
    print("Bag MLP:  " + results(targets_test, bag_mlp))
    print("ADA:      " + str(math.floor(ada(data, targets) * 100)) + "% Match")
    print("Forest:   " + str(math.floor(forest(data, targets) * 100)) + "% Match")


def predict_tree(data_train, data_test, targets_train):
    classifier = DecisionTreeClassifier()
    classifier.fit(data_train, targets_train)

    return classifier.predict(data_test)


def predict_knn(data_train, data_test, targets_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, targets_train)

    return model.predict(data_test)


def predict_mlp(data_train, data_test, targets_train):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), activation='logistic')
    clf.fit(data_train, targets_train)

    return clf.predict(data_test)


def bagging(data_train, data_test, targets_train):
    # Tree
    clf_tree = ensemble.BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
    clf_tree.fit(data_train, targets_train)

    # KNN
    clf_knn = ensemble.BaggingClassifier(KNeighborsClassifier(n_neighbors=3), max_samples=0.5, max_features=0.5)
    clf_knn.fit(data_train, targets_train)

    # MLP
    clf_mlp = ensemble.BaggingClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), activation='logistic'), max_samples=0.5, max_features=0.5)
    clf_mlp.fit(data_train, targets_train)

    return clf_tree.predict(data_test), clf_knn.predict(data_test), clf_mlp.predict(data_test)


def ada(data, targets):
    clf = ensemble.AdaBoostClassifier(n_estimators=50)
    scores = cross_val_score(clf, data, targets)

    return scores.mean()


def forest(data, targets):
    clf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=3, random_state=1)
    scores = cross_val_score(clf, data, targets)

    return scores.mean()


def main():
    load_lens()
    print()
    load_iris()
    print()
    load_abalone()


def results(targets, predictions):
    return str(math.floor(accuracy_score(predictions, targets) * 100)) + "% Match"


if __name__ == "__main__":
    main()
