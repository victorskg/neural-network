import numpy as np
import pandas as pd

class Perceptron(object):
    inputs_size = 0
    weights, values = [], []
    data_set, train_data, test_data = [], [], []

    def __init__(self, learn_rate=0.1, max_epochs=200, data_path="datasets/iris.data"):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.data_set = self._read_data(data_path)
        self.prepare_iris_data()

    def prepare_iris_data(self):
        self.data_set['class'] = self.data_set['class'].replace('Iris-setosa', '1')
        self.data_set['class'] = self.data_set['class'].replace('Iris-versicolor', '0')
        self.data_set['class'] = self.data_set['class'].replace('Iris-virginica', '0')

        self.data_set = self.data_set.to_numpy()

    def train(self, train_data, inputs):
        self.weights = np.random.rand(len(inputs) + 1)

        for _ in range(self.max_epochs):
            np.random.shuffle(train_data)
            for iris in train_data:
                data_in = self._get_inputs(iris, inputs)
                guess = self.classify(data_in, self.weights)
                error = int(iris[len(iris)-1]) - guess
                update = self.learn_rate * error
                self.weights[1:] += update * np.array(data_in)
                self.weights[0] += update 

    def test(self, test_data, inputs):
        hits = 0
        for iris in test_data:
            value = self.classify(self._get_inputs(iris, inputs), self.weights)
            hits = hits + 1 if value == int(iris[len(iris)-1]) else hits
            #print(self._get_inputs(iris, inputs))
            #print('Predict: {0}, Class: {1}'.format(value, int(iris[len(iris)-1])))
        accuracy = (hits / len(test_data)) * 100
        print(accuracy)
        return accuracy

    @staticmethod
    def classify(inputs, weights):
        value = np.dot(inputs, weights[1:]) + weights[0]
        return 1 if value >= 0.0 else 0

    @staticmethod
    def _read_data(path):
        data = pd.read_csv(path, header=None)
        data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

        return data

    @staticmethod
    def _get_inputs(row, inputs):
        return [row[inputs[i]] for i in range(len(inputs))]
