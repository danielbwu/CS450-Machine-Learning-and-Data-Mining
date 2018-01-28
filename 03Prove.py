import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math

# Get input from user
testSize = float(input("Enter test size (e.g, 0.3): "))
print(testSize)
print()


class MyClassifier:
    """ Returns a HardCodedModel """
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
        predicted = np.zeros(len(data_test), dtype=np.int)

        for i in range(len(data_test)):
            predicted[i] = self.targets[neighbors[i][0]]
        return predicted

    def knn(self, data_test):
        neighbors = [np.zeros(self.k, dtype=np.int) for x in range(len(data_test))]
        for x in range(len(data_test)):
            neighbors[x] = self.get_distances(data_test[x])

        return neighbors

    def get_distances(self, test):
        length = len(self.data)
        distance_map = {}
        distances = np.zeros(length, dtype=np.float)
        for i in range(length):
            distance = 0.0
            for j in range(len(test)):
                distance += pow((test[j] - self.data[i][j]), 2)
            distances[i] = float(math.sqrt(distance))
            distance_map[float(math.sqrt(distance))] = i

        distances = np.sort(distances)
        #print(distances)
        # print(distance_map)
        #print(self.k)
        nearest = np.zeros(self.k, dtype=np.int)
        for i in range(self.k):
            nearest[i] = distance_map[distances[i]]

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


def main():
    # Load Data
    iris = datasets.load_iris()

    # Split data into training and testing sets
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=testSize)

    # Use sk-learn KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)

    # Use custom classifier
    myClassifier = MyClassifier(3)
    myModel = myClassifier.fit(data_train, targets_train)
    myPredictions = myModel.predict(data_test)

    # Display Results
    print("Actual Results: ")
    print(targets_test)
    print()
    print("KNeighborsClassifier Results: " + str(calc_results(targets_test, targets_predicted)) + "% Match")
    print(targets_predicted)
    print()
    print("MyClassifier Results: " + str(calc_results(targets_test, myPredictions)) + "% Match")
    print(myPredictions)

    #print(myModel.neighbors)
    #print(data_train)
    #print(targets_train)
    #print(iris.target_names)


if __name__ == "__main__":
    main()
