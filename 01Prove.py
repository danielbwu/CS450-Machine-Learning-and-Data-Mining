import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

testSize = float(input("Enter test size (e.g, 0.3): "))
print(testSize)
print()


class HardCodedClassifier:
    """ Returns a HardCodedModel """
    def fit(self, data_train, targets_train):
        return HardCodedModel(data_train, targets_train)


class HardCodedModel:
    """ Constructor"""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    """ Predicts targets """
    def predict(self, data_test):
        predicted = np.zeros(len(data_test), dtype=np.int)
        return predicted


def calcResults(actual, test):
    """ Calculates % match"""
    match = 0
    size = len(actual)
    for i in range(size):
        if actual[i] == test[i]:
            match += 1

    return int(match / size * 100)


# Load Data
iris = datasets.load_iris()

# Split data into training and testing sets
data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=testSize)

# Create model
classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

# Make predictions
targets_predicted = model.predict(data_test)

# Make predictions with hard coded classes
myClassifier = HardCodedClassifier()
myModel = myClassifier.fit(data_train, targets_train)
myPredictions = myModel.predict(data_test)

# Display Results
print("Actual Results: ")
print(targets_test)
print()
print("Sk-Learn Results: " + str(calcResults(targets_test, targets_predicted)) + "% Match")
print(targets_predicted)
print()
print("Hard-Coded Results: " + str(calcResults(targets_test, myPredictions)) + "% Match")
print(myPredictions)
