import numpy as np
import matplotlib.pyplot as plt
from models.Perceptron import Perceptron

def main():
    perceptron = Perceptron(learn_rate = 0.1, max_epochs = 30, data_path = "datasets/iris.data")

    def _calc_accuracy(array):
        return sum(array) / len(array)

    def print_points(inputs):
        p = perceptron.weights;
        actual_type, others = [], [] 
        for i in range(len(perceptron.values)): 
            if (perceptron.test_data[i].expected_type == 1):
                actual_type.append(perceptron.test_data[i].inputs[:2])
            else: others.append(perceptron.test_data[i].inputs[:2])
        
        # a = np.linspace(0 ,10,100)
        # b = (-(p[0]/p[2]) / (p[0]/p[1]))*a + (-p[0] / p[2])
        # plt.plot(a, b, '-r')

        x, y = np.array(actual_type).T
        w, z = np.array(others).T
        plt.xlabel(inputs[0])
        plt.ylabel(inputs[1])
        plt.plot(x,y, 'bo', w, z, 'go')
        plt.show()

    # def decision_surface(inputs, weigths):
    #     x, y = 0, 0
    #     actual_type, others = [], [] 
    #     while(x <= 7.0):
    #         y = 0
    #         while(y <= 7.0):
    #             if (perceptron.classify([x, y], weigths) == 1): actual_type.append([x, y])
    #             else: others.append([x, y])
    #             y += 0.2
    #         x += 0.2
    #     z, k = np.array(actual_type).T
    #     w, z = np.array(others).T
    #     plt.xlabel(inputs[0])
    #     plt.ylabel(inputs[1])
    #     plt.plot(z, k, 'bo')
    #     plt.show()

    def setosa4INPUTS():
        max_acuracy = 0
        selected_matrix = []
        accuracys = []
        inputs = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        perceptron.prepare_data("Iris-setosa", inputs)
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy, matrix = perceptron.accuracy_and_matrix()
            if (accuracy > max_acuracy):
                max_acuracy = accuracy
                selected_matrix = matrix
            accuracys.append(accuracy)
            print('Realização: {0}, Acurácia: {1}%'.format(i, accuracy))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys), np.std(accuracys)))
        #print('Matrix de confusão:')
        #print(selected_matrix)
        print_points(inputs)

    #setosa4INPUTS()

    def versicolor4INPUTS():
        max_acuracy = 0
        rel = 0
        selected_matrix = []
        accuracys = []
        inputs = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        perceptron.prepare_data("Iris-versicolor", inputs)
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy, matrix = perceptron.accuracy_and_matrix()
            if (accuracy > max_acuracy):
                max_acuracy = accuracy
                rel = i
                selected_matrix = matrix
            accuracys.append(accuracy)
            print('Realização: {0}, Acurácia: {1}%'.format(i, accuracy))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys), np.std(accuracys)))
        #print('Matrix de confusão da realização ', rel)
        #print(selected_matrix)
        print_points(inputs)

    #versicolor4INPUTS()

    def virginica4INPUTS():
        max_acuracy = 0
        rel = 0
        selected_matrix = []
        accuracys = []
        inputs = ['sepal_width', 'petal_width']
        perceptron.prepare_data("Iris-virginica", inputs)
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy, matrix = perceptron.accuracy_and_matrix()
            if (accuracy > max_acuracy):
                max_acuracy = accuracy
                rel = i
                selected_matrix = matrix
            accuracys.append(accuracy)
            print('Realização: {0}, Acurácia: {1}%'.format(i, accuracy))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys), np.std(accuracys)))
        #print('Matrix de confusão da realização ', rel)
        #print(selected_matrix)
        print_points(inputs)

    virginica4INPUTS()

if __name__ == '__main__':
    main()
    