import numpy as np
import pandas as pd

from .Iris import Iris


class Perceptron(object):
    learn_rate = 0.0
    max_epochs = 0
    weights = []
    data_set, iris_data, train_data, test_data = [], [], [], []

    def __init__(self, learn_rate=0.1, max_epochs=20, data_path="datasets/iris.data", data_type="Iris-setosa"):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.data_set = self._read_data(data_path)
        self.prepare_data(data_type)

    @staticmethod
    def _read_data(path):
        data = pd.read_csv(path, header=None)
        data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

        return data

    @staticmethod
    def _get_inputs(row):
        return [row['sepal_length'], row['sepal_width'], row['petal_length'],row['petal_width'], row['class']]

    def prepare_data(self, data_type):
        qt_trainning = int(0.8 * len(self.data_set))
        for row in self.data_set.iterrows():
            self.iris_data.append(Iris(inputs=self._get_inputs(row=row[1]), expected_type=data_type))
            np.random.shuffle(self.iris_data)

        self.train_data, self.test_data = self.iris_data[:qt_trainning], self.iris_data[qt_trainning:]

    def train(self):
        self.weights = np.random.rand(5)
        for epoch in range(self.max_epochs):
            np.random.shuffle(self.train_data)
            #print('Epoca: {0}, Pesos: {1}'.format(epoch, self.weights))
            for iris in self.test_data:
                guess = self.classify(iris.inputs[:4])
                error = iris.expected_type - guess
                self.weights[1:] += self.learn_rate * error * np.array(iris.inputs[:4])
                self.weights[0] += self.learn_rate * error * 1

    def classify(self, inputs):
        value = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if value >= 0.0 else -1

    def test(self):
        for iris in self.test_data:
            value = self.classify(iris.inputs[:4])
            print('Guess_Type: {0}, Expected_Type: {1}'.format(value, iris.inputs[4]))
