import numpy as np
import matplotlib.pyplot as plt
from models.Perceptron import Perceptron

def main():
    perceptron = Perceptron(learn_rate = 0.1, max_epochs = 20, data_path = "datasets/iris.data")

    def _calc_accuracy(array):
        return sum(array) / len(array)

    def setosa4INPUTS():
        accuracys = []
        perceptron.prepare_data("Iris-setosa", ['petal_length', 'petal_width'])
        for i in range(20):
            perceptron.train()
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            #print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))
        actual_type, others = [], [] 
        for i in range(len(perceptron.values)): 
            if (perceptron.test_data[i].expected_type == 1):
                actual_type.append(perceptron.test_data[i].inputs[:2])
            else: others.append(perceptron.test_data[i].inputs[:2])
        
        x, y = np.array(actual_type).T
        w, z = np.array(others).T
        print(actual_type)
        plt.plot(x,y, 'bo', w, z, 'go')
        plt.show()

    setosa4INPUTS()

    def versicolor4INPUTS():
        accuracys = []
        perceptron.prepare_data("Iris-versicolor", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        for i in range(20):
            perceptron.train()
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))

    #versicolor4INPUTS()

    def virginica4INPUTS():
        accuracys = []
        perceptron.prepare_data("Iris-virginica", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        for i in range(20):
            perceptron.train()
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))

    #virginica4INPUTS()
    
    

if __name__ == '__main__':
    main()
    