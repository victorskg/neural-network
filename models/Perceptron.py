import numpy as np
import pandas as pd

from .Iris import Iris


class Perceptron(object):
    max_epochs = 0
    inputs_size = 0
    learn_rate = 0.0
    weights, values = [], []
    data_set, iris_data, train_data, test_data = [], [], [], []

    def __init__(self, learn_rate=0.1, max_epochs=30, data_path="datasets/iris.data"):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.data_set = self._read_data(data_path)

    @staticmethod
    def _read_data(path):
        data = pd.read_csv(path, header=None)
        data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

        return data

    def _get_inputs(self, row, inputs_str):
        inputs = []
        for in_str in inputs_str:
            inputs.append(row[in_str])
        inputs.append(row['class'])

        return inputs

    def prepare_data(self, data_type, inputs_str):
        self.inputs_size = len(inputs_str)
        self.iris_data, self.train_data, self.test_data = [], [], []
        for row in self.data_set.iterrows():
            self.iris_data.append(Iris(inputs=self._get_inputs(row[1], inputs_str), expected_type=data_type))
            np.random.shuffle(self.iris_data)

    def train(self, dataset):
        qt_trainning = int(0.8 * len(dataset))
        self.weights = np.random.rand(self.inputs_size + 1)
        self.train_data, self.test_data = dataset[:qt_trainning], dataset[qt_trainning:]

        for epoch in range(self.max_epochs):
            np.random.shuffle(self.train_data)
            #print('Epoca: {0}, Pesos: {1}'.format(epoch, self.weights))
            for iris in self.test_data:
                guess = self.classify(iris.inputs[:self.inputs_size])
                error = iris.expected_type - guess
                update = self.learn_rate * error
                self.weights[1:] += update * np.array(iris.inputs[:self.inputs_size])
                self.weights[0] += update 

    def classify(self, inputs):
        value = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if value >= 0.0 else -1

    def test(self):
        self.values = []
        for iris in self.test_data:
            value = self.classify(iris.inputs[:self.inputs_size])
            self.values.append(value)
            #print('Guess_Type: {0}, Expected_Type: {1}'.format(value, iris.inputs[self.inputs_size]))

    def accuracy(self):
        hits = 0
        for xi, yi in zip(self.test_data, self.values):
            if (xi.expected_type == yi): hits += 1
        
        accuracy = (hits / len(self.test_data)) * 100
        return accuracy
