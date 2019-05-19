import numpy as np
import pandas as pd
from models.Perceptron import Perceptron

class SingleLayerPerceptron(Perceptron):
    def __init__(self, learn_rate=0.1, max_epochs=200, data_path="datasets/iris.data"):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.data_set = self._read_data(data_path)
        self.define_class()

    def define_class(self):
        self.data_set['class'] = self.data_set['class'].replace('Iris-setosa', '100')
        self.data_set['class'] = self.data_set['class'].replace('Iris-versicolor', '010')
        self.data_set['class'] = self.data_set['class'].replace('Iris-virginica', '001')
    
    def prepare_data(self):
        self.data_set = self.data_set.to_numpy()
        np.random.shuffle(self.data_set)
        
    def train(self, inputs):
        qt_trainning = int(0.8 * len(self.data_set))
        self.train_data, self.test_data = self.data_set[:qt_trainning], self.data_set[qt_trainning:]
        self.weights_1 = np.random.rand(len(inputs) + 1)
        self.weights_2 = np.random.rand(len(inputs) + 1)
        self.weights_3 = np.random.rand(len(inputs) + 1)
        for _ in range(self.max_epochs):
            np.random.shuffle(self.train_data)
            for iris in self.train_data:
                expected = np.array(list(iris[4])).astype(np.int)
                selected_inputs = [iris[inputs[i]] for i in range(len(inputs))]
                self.run_first_neuron(selected_inputs, expected[0])
                self.run_second_neuron(selected_inputs, expected[1])
                self.run_third_neuron(selected_inputs, expected[2])

    def run_first_neuron(self, inputs, expected):
        self.train_update(self.weights_1, inputs, expected)

    def run_second_neuron(self, inputs, expected):
        self.train_update(self.weights_2, inputs, expected)

    def run_third_neuron(self, inputs, expected):
        self.train_update(self.weights_3, inputs, expected)

    def train_update(self, weights, inputs, expected):
        guess = self.classify(inputs, weights)
        error = expected - guess
        update = self.learn_rate * error
        weights[1:] += update * np.array(inputs)
        weights[0] += update 
    
    def test(self, inputs):
        for iris in self.test_data:
            guess = []
            selected_inputs = [iris[inputs[i]] for i in range(len(inputs))]
            expected = np.array(list(iris[4])).astype(np.int)
            guess.append(self.classify(selected_inputs, self.weights_1))
            guess.append(self.classify(selected_inputs, self.weights_2))
            guess.append(self.classify(selected_inputs, self.weights_3))
            print('Expected: {0}, Guess: {1}'.format(expected, guess))
