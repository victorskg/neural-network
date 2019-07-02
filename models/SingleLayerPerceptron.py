import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.Perceptron import Perceptron


class SingleLayerPerceptron(Perceptron):
    def __init__(self, learn_rate=0.1, neuron_count=3, max_epochs=200, data_path="datasets/iris.data"):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.neuron_count = neuron_count
        self.data_set = self._read_data(data_path)
        self.define_class()

    def define_class(self):
        self.data_set['class'] = self.data_set['class'].replace('Iris-setosa', '100')
        self.data_set['class'] = self.data_set['class'].replace('Iris-versicolor', '010')
        self.data_set['class'] = self.data_set['class'].replace('Iris-virginica', '001')

    def prepare_data(self, activation):
        self.activation = activation
        self.data_set = self.data_set.to_numpy()

    def generate_artificial_data(self, activation):
        self.activation = activation
        data = self.create_points([1, 0], '010')
        data = data.append(self.create_points([0, 0], '100'), ignore_index=True)
        data = data.append(self.create_points([1, 1], '001'), ignore_index=True)
        self.artiticial_data = data.to_numpy()

    def predict(self, x):
        if self.activation == 0:
            return self.classify(x)
        elif self.activation == 1:
            return self.logistic_sigmoid(x)
        else:
            return self.hyperbolic_tangent(x)

    def derivative(self, x):
        if self.activation == 0:
            return 1
        elif self.activation == 1:
            return self.logistic_sigmoid(x) * (1 - self.logistic_sigmoid(x))
        else:
            return 0.5 * (1 - self.hyperbolic_tangent(x) * self.hyperbolic_tangent(x))

    def update_weight(self, weights, inputs, error, x):
        update = self.learn_rate * error * self.derivative(x)
        weights[1:] += update * np.array(inputs)
        weights[0] += update

    def train(self, inputs):
        self.weights = [np.random.rand(len(inputs) + 1) for _ in range(self.neuron_count)]
        for _ in range(self.max_epochs):
            np.random.shuffle(self.train_data)
            for data in self.train_data:
                outputs, guess = [], []
                expected = np.array(list(data[len(data) - 1])).astype(np.int)
                selected_inputs = [data[inputs[i]] for i in range(len(inputs))]
                for i in range(self.neuron_count):
                    outputs.append(self.output(selected_inputs, self.weights[i]))
                    guess.append(self.predict(outputs[i]))
                guess = self.validate_guess(guess, outputs) if self.activation == 0 else guess
                for i in range(self.neuron_count):
                    error = expected[i] - guess[i]
                    self.update_weight(self.weights[i], selected_inputs, error, guess[i])

    def test(self, inputs):
        hits = 0
        for data in self.test_data:
            outputs, guess = [], []
            expected = np.array(list(data[len(data) - 1])).astype(np.int)
            selected_inputs = [data[inputs[i]] for i in range(len(inputs))]
            for i in range(self.neuron_count):
                outputs.append(self.output(selected_inputs, self.weights[i]))
                guess.append(self.predict(outputs[i]))
            guess = self.validate_guess(guess, outputs)
            hits = hits + 1 if np.array_equal(guess, expected) else hits
            # print('Expected: {0}, Guess: {1}'.format(expected, guess))

        return (hits / len(self.test_data)) * 100

    def plot_decision_surface(self, inputs):
        x1_colunm, x2_colunm = self.test_data[:, inputs[0]], self.test_data[:, inputs[1]]
        x1_max, x1_min = np.amax(x1_colunm) + 0.5, np.amin(x1_colunm) - 0.5
        x2_max, x2_min = np.amax(x2_colunm) + 0.5, np.amin(x2_colunm) - 0.5

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.07), np.arange(x2_min, x2_max, 0.07))
        Z = np.array([xx1.ravel(), xx2.ravel()]).T

        fig, ax = plt.subplots()
        ax.set_facecolor((0.97, 0.97, 0.97))
        for x1, x2 in Z:
            outputs, guess = [], []
            for i in range(self.neuron_count):
                outputs.append(self.output([x1, x2], self.weights[i]))
                guess.append(self.predict(outputs[i]))
            guess = self.validate_guess(guess, outputs)
            if np.array_equal(guess, [1, 0, 0]):
                ax.scatter(x1, x2, c='red', s=1.5, marker='o')
            elif np.array_equal(guess, [0, 1, 0]):
                ax.scatter(x1, x2, c='green', s=1.5, marker='o')
            elif np.array_equal(guess, [0, 0, 1]):
                ax.scatter(x1, x2, c='blue', s=1.5, marker='o')

        for row in self.test_data:
            expected = np.array(list(row[len(row) - 1])).astype(np.int)
            if np.array_equal(expected, [1, 0, 0]):
                ax.scatter(row[inputs[0]], row[inputs[1]], c='red', marker='v')
            elif np.array_equal(expected, [0, 1, 0]):
                ax.scatter(row[inputs[0]], row[inputs[1]], c='green', marker='*')
            elif np.array_equal(expected, [0, 0, 1]):
                ax.scatter(row[inputs[0]], row[inputs[1]], c='blue', marker='o')
        plt.show()

    def divide_data(self, dataset):
        np.random.shuffle(dataset)
        qt_training = int(0.8 * len(dataset))
        self.train_data, self.test_data = dataset[:qt_training], dataset[qt_training:]

    @staticmethod
    def output(inputs, weights):
        return np.dot(inputs, weights[1:]) + weights[0]

    @staticmethod
    def classify(x):
        return 1.0 if x >= 0.0 else 0.0

    @staticmethod
    def logistic_sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def hyperbolic_tangent(x):
        return math.tanh(x)

    @staticmethod
    def validate_guess(guess, outputs):
        new_guess = guess
        if np.sum(guess) != 1:
            new_guess = [1 if output == np.amax(outputs) else 0 for output in outputs]
        return new_guess

    @staticmethod
    def create_points(source, _class):
        points = []
        for _ in range(50):
            coords = [source[i] + np.random.random() * 0.09 for i in range(2)]
            coords.append(_class)
            points.append(coords)
        return pd.DataFrame(data=points, columns=['x1', 'x2', 'd'])
