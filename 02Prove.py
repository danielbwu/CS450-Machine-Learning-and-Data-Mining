import numpy as np
from sklearn import datasets
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
        self.neighbors = knn(data, k)
        self.k = k

    """ Predicts targets """
    def predict(self, data_test):
        predicted = np.zeros(len(data_test), dtype=np.int)
        return predicted


def calc_results(actual, test):
    """ Calculates % match """
    match = 0
    size = len(actual)
    for i in range(size):
        if actual[i] == test[i]:
            match += 1

    return int(match / size * 100)


def get_distances(x, data):
    length = len(data[0])
    distances = {}
    for i in range(len(data)):
        distance = 0.0
        if i != x:
            for j in range(length):
                distance += pow((data[x][j] - data[i][j]), 2)
            distances[i] = float(math.sqrt(distance))
        else:
            distances[x] = 0.0;
    return distances


def knn(data_train, k):
    neighbors = [dict() for x in range(len(data_train))]
    for x in range(len(data_train)):
        neighbors[x] = get_distances(x, data_train)
    return neighbors


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

print(myModel.neighbors)
