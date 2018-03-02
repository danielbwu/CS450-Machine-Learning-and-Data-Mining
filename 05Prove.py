import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Node:

    def __init__(self):
        self.weights = []
        self.bias = 0


class Layer:

    def __init__(self):
        self.nodes = [Node() for x in range(4)]


class Network:

    def __init__(self):
        self.layers = []


def main():

    node = Node()


