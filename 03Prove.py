import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import math
from statistics import mode

# Get input from user
# testSize = float(input("Enter test size (e.g, 0.3): "))
# print(testSize)
# print()
nn = 3


class MyClassifier:
    """ Constructor """
    def __init__(self, k):
        self.k = k

    def fit(self, data_train, targets_train):
        return MyModel(data_train, targets_train, self.k)


class MyModel:
    """ Constructor """
    def __init__(self, data, targets, k):
        self.data = data
        self.targets = targets
        self.k = k

    """ Predicts targets """
    def predict(self, data_test):
        neighbors = self.knn(data_test)
        # predicted = np.zeros(len(data_test), dtype=np.int)
        predicted = []

        for i in range(len(data_test)):
            m = None
            try:
                m = mode(neighbors[i])
            except ValueError:
                m = neighbors[i][0]
            predicted.append(self.targets[m])
        return predicted

    def knn(self, data_test):
        #neighbors = [[] for x in range(len(data_test))]
        neighbors = []
        for x in range(len(data_test)):
            neighbors.append(self.get_distances(data_test[x]))

        return neighbors

    def get_distances(self, test):
        length = len(self.data)
        distance_map = {}
        #distances = np.zeros(length, dtype=np.float)
        distances = []
        for i in range(length):
            distance = 0.0
            for j in range(len(test)):
                distance += pow((float(test[j]) - float(self.data[i][j])), 2)
            distance = float(math.sqrt(distance))
            distances.append(distance)
            distance_map[distance] = i

        #distances = np.sort(distances)
        distances.sort()
        #print(distances)
        # print(distance_map)
        #print(self.k)

        #nearest = np.zeros(self.k, dtype=np.int)
        nearest = []
        for i in range(self.k):
            nearest.append(distance_map[distances[i]])

        #print(nearest)
        return nearest


def calc_results(actual, test):
    """ Calculates % match """
    match = 0
    size = len(actual)
    for i in range(size):
        if actual[i] == test[i]:
            match += 1

    return int(match / size * 100)


def cars():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    cars = pd.read_csv("car.data.txt", header=None, names=headers, na_values="?")

    cleanup = {"buying": {"vhigh": 4.0, "high": 3.0, "med": 2.0, "low": 1.0},
               "maint":  {"vhigh": 4.0, "high": 3.0, "med": 2.0, "low": 1.0},
               "doors": {"5more": 5.0},
               "persons": {"more": 5.0},
               "lug_boot": {"small": 1.0, "med": 2.0, "big": 3.0},
               "safety": {"high": 3.0, "med": 2.0, "low": 1.0},
               "class": {"unacc": 1, "acc": 2, "good": 3, "vgood": 4}}
    cars.replace(cleanup, inplace=True)

    cars_target = cars["class"]
    cars_data = cars.drop("class", axis=1)
    # print(cars_data)

    # Train and predict
    return cars_data, cars_target


def mpg():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin", "name"]
    mpg = pd.read_csv("auto-mpg.data.txt", header=None, names=headers, na_values="?", delim_whitespace=True)

    # Clean up data
    mpg = mpg.dropna(axis=0, how='any')
    mpg_target = mpg["mpg"]
    mpg_data = mpg.drop(["mpg", "name"], axis=1)
    #print(mpg_data)

    # Train and predict
    # print("MPG: ")
    # test(mpg_data, mpg_target)
    #print(mpg_data.values.tolist())

    return mpg_data, mpg_target


def test(data, targets):
    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.3)

    # Use sk-learn KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=nn)
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    print("Sklearn Results: " + str(calc_results(targets_test, targets_predicted)) + "% Match")

    # Use custom classifier
    myClassifier = MyClassifier(nn)
    myModel = myClassifier.fit(data_train, targets_train)
    myPredictions = myModel.predict(data_test)

    # Display results
    print("My Results: " + str(calc_results(targets_test, myPredictions)) + "% Match")
    #print(targets_predicted)
    #print(targets_test.data)
    #print(myPredictions)


def main():
    print("Cars: ")
    cars_data, cars_target = cars()
    test(cars_data.values.tolist(), cars_target.values.tolist())
    print()
    print("MPG: ")
    mpg_data, mpg_target = mpg()

    data_train, data_test, targets_train, targets_test = train_test_split(mpg_data.values.tolist(), mpg_target.values.tolist(), test_size=0.3)
    # Use custom classifier
    myClassifier = MyClassifier(nn)
    myModel = myClassifier.fit(data_train, targets_train)
    myPredictions = myModel.predict(data_test)
    print("My Results: " + str(calc_results(targets_test, myPredictions)) + "% Match")
    # print(myPredictions)

    # Sklearn regressor
    reg = KNeighborsRegressor(n_neighbors=nn)
    reg.fit(data_train, targets_train)
    reg_results = reg.predict(data_test)
    print("Sklearn Results: " + str(calc_results(targets_test, reg_results)) + "% Match")


if __name__ == "__main__":
    main()
