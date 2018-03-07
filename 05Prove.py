import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import random
import math


class Node:
    """Represents a single neuron in a network"""
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.sumWeight = 0
        self.activation = 0
        self.error = 0

    """Sets random values to weights"""
    def set_weights(self, size):
        self.weights = [random.uniform(-1.0, 1.0) for x in range(size)]

    """Activation function"""
    def activate(self, inputs, bias):
        self.inputs = inputs
        self.sumWeight = sum_weights(inputs, self.weights, bias)
        self.activation = 1 / (1 + math.exp(-1 * self.sumWeight))

        return self.activation

    """Adjusts weights"""
    def adjust_weights(self, bias, learn_rate):
        self.weights[0] = update_weight(self.weights[0], learn_rate, self.error, bias)
        for i in range(1, len(self.weights)):
            self.weights[i] = update_weight(self.weights[i], learn_rate, self.error, self.inputs[i - 1])


class Layer:
    """Represents a layer of nodes in a perceptron"""
    def __init__(self):
        self.size = random.randint(2, 4)
        self.nodes = [Node() for x in range(self.size)]

    """Initializes default weights"""
    def set_weights(self, size):
        for i in range(len(self.nodes)):
            self.nodes[i].set_weights(size)

    """Returns activation values for all nodes in layer"""
    def feed_forward(self, inputs, bias):
        outputs = []
        for i in range(self.size):
            outputs.append(self.nodes[i].activate(inputs, bias))

        return outputs

    """Calls each Node to update its weights"""
    def adjust_weights(self, bias, learn_rate):
        for i in range(self.size):
            self.nodes[i].adjust_weights(bias, learn_rate)


class Network:
    """Represents a neural network"""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.layers = [Layer() for x in range(2)]
        self.bias = -1
        self.learn_rate = 0.05
        self.init_weights()

    # Initializes weights to random values
    def init_weights(self):
        self.layers[0].set_weights(len(self.data[0]) + 1)
        for i in range(1, len(self.layers)):
            self.layers[i].set_weights(self.layers[i - 1].size + 1)

    # Handles feed forward and back propagation
    def train(self, inputs, data_index, layer_index):
        # Send input to activate nodes in current layer
        output = self.layers[layer_index].feed_forward(inputs, self.bias)

        # Recursively activate each layer
        if (layer_index + 1) < len(self.layers):
            self.train(output, data_index, layer_index + 1)

            # Back-prop for hidden layer
            k = self.layers[layer_index + 1].nodes # Next layer
            for i in range(self.layers[layer_index].size):
                errorSum = 0
                for j in range(len(k)):
                    errorSum += k[j].error * k[j].weights[i + 1]

                a = self.layers[layer_index].nodes[i].activation
                self.layers[layer_index].nodes[i].error = a * (1 - a) * errorSum
        else:
            # Back-prop for output layer
            for i in range(self.layers[layer_index].size):
                a = self.layers[layer_index].nodes[i].activation
                self.layers[layer_index].nodes[i].error = a * (1 - a) * (a - self.targets[data_index])

    # Adjusts weights for each node
    def adjust_weights(self):
        # Loop through each layer to update Node weights
        for i in range(len(self.layers)):
            self.layers[i].adjust_weights(self.bias, self.learn_rate)

    # Trains network
    def fit(self):
        for i in range(len(self.data)):
            self.train(self.data[i], i, 0)
            self.adjust_weights()

    # Predicts output
    def predict(self, data):
        output = [0.0 for x in range(len(data))]
        for i in range(len(data)):
            inputs = np.array(data[i])
            for j in range(len(self.layers)):
                inputs = np.array(self.layers[j].feed_forward(inputs, self.bias))
            output[i] = np.average(inputs)
            # print(inputs)

        return output


# Calculates the sum of inputs * weights
def sum_weights(inputs, weights, bias):
    weightSum = 0.0
    weightSum += bias * weights[0]
    for i in range(1, len(inputs)):
        weightSum += inputs[i] * weights[i]

    return weightSum


# Performs the weight update calculation
def update_weight(weight, rate, error, input):
    return weight - (rate * error * input)


# Calculates node activation
def activate(x):
    return 1 / (1 + math.exp(-1 * x))


def main():
    # Load Data
    iris = datasets.load_iris()
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.3)

    # Use sklearn MLP
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), activation='logistic')
    clf.fit(data_train, targets_train)
    predictions = clf.predict(data_test)

    # Custom Network
    network = Network(data_train, targets_train)
    network.fit()
    my_predictions = network.predict(data_test)

    # Print Results
    print("Actual Results: ")
    print(targets_test)
    print()
    print("Sklearn Results: " + str(math.floor(accuracy_score(predictions, targets_test) * 100)) + "% Match")
    print(predictions)
    print()
    print("My Network Results: ")
    print(my_predictions)

    #print(accuracy_score(predictions, targets_test))


if __name__ == "__main__":
    main()
