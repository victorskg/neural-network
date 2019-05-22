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
    
    def prepare_data(self):
        self.data_set = self.data_set.to_numpy()
        
    def train(self, inputs):
        np.random.shuffle(self.data_set)
        qt_trainning = int(0.8 * len(self.data_set))
        self.train_data, self.test_data = self.data_set[:qt_trainning], self.data_set[qt_trainning:]
        self.weights = [np.random.rand(len(inputs) + 1) for _ in range(self.neuron_count)] 
        for _ in range(self.max_epochs):
            np.random.shuffle(self.train_data)        
            for iris in self.train_data:
                outputs, guess = [], []
                expected = np.array(list(iris[4])).astype(np.int)
                selected_inputs = [iris[inputs[i]] for i in range(len(inputs))]       
                for i in range(self.neuron_count):
                    outputs.append(self.output(selected_inputs, self.weights[i]))
                    guess.append(self.classify(outputs[i])) 
                high_output = np.amax(outputs)
                guess = self.validate_guess(guess, high_output, outputs)
                for i in range(self.neuron_count):
                    error = expected[i] - guess[i]
                    self.update_weight(self.weights[i], selected_inputs, error)                

    def update_weight(self, weights, inputs, error):
        update = self.learn_rate * error
        weights[1:] += update * np.array(inputs)
        weights[0] += update 
    
    def test(self, inputs):
        hits = 0
        for iris in self.test_data:
            outputs, guess = [], []
            selected_inputs = [iris[inputs[i]] for i in range(len(inputs))]
            expected = np.array(list(iris[4])).astype(np.int)
            for i in range(self.neuron_count):
                outputs.append(self.output(selected_inputs, self.weights[i]))
                guess.append(self.classify(outputs[i])) 
            high_output = np.amax(outputs)
            guess = self.validate_guess(guess, high_output, outputs)
            hits = hits + 1 if np.array_equal(guess, expected) else hits
            #print('Expected: {0}, Guess: {1}'.format(expected, guess))

        return (hits / len(self.test_data)) * 100
    
    @staticmethod
    def output(inputs, weights):
        return np.dot(inputs, weights[1:]) + weights[0]
    
    @staticmethod
    def classify(value):
        return 1 if value >= 0.0 else 0

    @staticmethod
    def validate_guess(guess, high_output, outputs):
        new_guess = guess
        if(np.sum(guess) != 1):
            new_guess = [1 if output == high_output else 0 for output in outputs]
        return new_guess
