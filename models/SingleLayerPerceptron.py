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
        self.data_set['class'] = self.data_set['class'].replace('Iris-setosa', '[1, 0, 0]')
        self.data_set['class'] = self.data_set['class'].replace('Iris-versicolor', '[0, 1, 0]')
        self.data_set['class'] = self.data_set['class'].replace('Iris-virginica', '[0, 0, 1]')
        print(self.data_set['class'])
        

    
