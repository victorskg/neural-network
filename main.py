import numpy as np
import matplotlib.pyplot as plt
from models.Perceptron import Perceptron
from models.Artificial import Artificial

def main():
    perceptron = Perceptron(learn_rate = 0.1, max_epochs = 20, data_path = "datasets/iris.data")

    def _calc_accuracy(array):
        return sum(array) / len(array)

    def print_points():
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

    def setosa4INPUTS():
        accuracys = []
        perceptron.prepare_data("Iris-setosa", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            #print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))
        print_points()

    #setosa4INPUTS()

    def versicolor4INPUTS():
        accuracys = []
        perceptron.prepare_data("Iris-versicolor", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            #print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))
        print_points()

    #versicolor4INPUTS()

    def virginica4INPUTS():
        accuracys = []
        perceptron.prepare_data("Iris-virginica", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            #print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))
        print_points()

    #virginica4INPUTS()

    def artificial_data():
        dataset = np.array([Artificial([np.random.uniform(0, 0.5), y], -1) for y in np.random.uniform(0, 0.5, 10)])
        dataset = np.append(dataset, [Artificial([np.random.uniform(0, 0.5), y], -1) for y in np.random.uniform(7, 7.5, 10)], axis=0)
        dataset = np.append(dataset, [Artificial([np.random.uniform(3, 3.5), y], -1) for y in np.random.uniform(0, 0.5, 10)], axis=0)
        #plt.plot(dataset[:, 0], dataset[:, 1], 'ro')
        dataset = np.append(dataset, [Artificial([np.random.uniform(3, 3.5), y], 1) for y in np.random.uniform(7, 7.5, 10)], axis=0)
        #plt.plot(dataset[30:, 0], dataset[30:, 1], 'bo')
        print(dataset)
        #plt.axis([-1, 4, -1, 8])
        #plt.show()
        np.random.shuffle(dataset)
        accuracys = []
        for i in range(20):
            perceptron.train(dataset)
            perceptron.test()
            accuracy = perceptron.accuracy()
            accuracys.append(accuracy)
            #print('Rel: {0}, Accuracy: {1}%'.format(i, _calc_accuracy(accuracys)))

        print('Final accuracy: {0}%'.format(_calc_accuracy(accuracys)))
    
    artificial_data()
    

if __name__ == '__main__':
    main()
    