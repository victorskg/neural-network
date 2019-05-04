import sys
from models.Perceptron import Perceptron

def main():
    perceptron = Perceptron(learn_rate = 0.1, max_epochs = 20, data_path = "datasets/iris.data", data_type = "Iris-setosa")
    perceptron.train()
    perceptron.test()

if __name__ == '__main__':
    main()
    