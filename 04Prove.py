import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import math


class TreeClassifier:

    def __init__(self):
        self.data = []
        self.targets = []

    def fit(self, data, targets):
        self.data = data
        self.targets = targets
        for i in range(4):
            self.calc_entropy(data.values[:, i])

    def calc_entropy(self, data):
        _, val_freqs = np.unique(data, return_counts=True)
        print(val_freqs)


def main():
    # Load Lens Data
    lense_type = {1: "hard", 2: "soft", 3: "none"}
    headers = ["age", "prescription", "astigmatic", "tear_rate", "lens"]
    lenses = pd.read_csv("lenses.data.txt", header=None, names=headers, na_values="?", delim_whitespace=True, index_col=0)
    lenses.head()

    lens_target = lenses["lens"]
    lens_data = lenses.drop("lens", axis=1)
    # print(lens_target)
    # print(lens_data)

    data_train, data_test, targets_train, targets_test = train_test_split(lens_data, lens_target, test_size=0.3)

    # print(lenses)
    # print(lenses.values[:, 4])
    # print(lenses.ix[:, 4])

    # Use sk-learn tree classifier
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(data_train, targets_train)
    targets_predict = classifier.predict(data_test)

    # Use Custom Tree Classifier
    my_classifier = TreeClassifier()
    my_classifier.fit(data_train, targets_train)

    print(targets_predict)
    print(targets_test.values)
    print(accuracy_score(targets_predict, targets_test.values))


if __name__ == "__main__":
    main()
