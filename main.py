import matplotlib.pyplot as plt
import numpy as np

from models.Adaline import Adaline
from models.Perceptron import Perceptron
from models.SingleLayerPerceptron import SingleLayerPerceptron


def main():
    adaline = Adaline(learn_rate=0.1, max_epochs=200)
    perceptron = Perceptron(learn_rate=0.1, max_epochs=200, data_path="datasets/iris.data")
    single_layer_perceptron = SingleLayerPerceptron(learn_rate=0.1, neuron_count=3, max_epochs=500,
                                                    data_path="datasets/iris.data")

    def _calc_accuracy(array):
        return sum(array) / len(array)

    def print_points(inputs, w, data):
        actual_type, others = [], []
        for i in range(len(perceptron.values)):
            if (data[i].expected_type == 1):
                actual_type.append(data[i].inputs[:2])
            else:
                others.append(data[i].inputs[:2])

        a = np.linspace(0, 10, 100)
        b = (-(w[0] / w[2]) / (w[0] / w[1])) * a + (-w[0] / w[2])
        plt.plot(a, b, '-r')

        x, y = np.array(actual_type).T
        w, z = np.array(others).T
        plt.xlabel(inputs[0])
        plt.ylabel(inputs[1])
        plt.plot(x, y, 'bo', w, z, 'go')
        plt.show()

    def setosa4INPUTS():
        max_acuracy = 0
        selected_matrix = []
        accuracys = []
        test_data, w = [], []
        inputs = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        perceptron.prepare_data("Iris-setosa", inputs)
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy, matrix = perceptron.accuracy_and_matrix()
            if (accuracy > max_acuracy):
                max_acuracy = accuracy
                selected_matrix = matrix
                test_data = perceptron.test_data
                w = perceptron.weights
            accuracys.append(accuracy)
            print('Realização: {0}, Acurácia: {1}%'.format(i, accuracy))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys),
                                                                              np.std(accuracys)))
        print('Matrix de confusão:')
        print(selected_matrix)
        # print_points(inputs, w, test_data)

    # setosa4INPUTS()

    def versicolor4INPUTS():
        max_acuracy = 0
        rel = 0
        selected_matrix = []
        accuracys = []
        test_data, w = [], []
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
                test_data = perceptron.test_data
                w = perceptron.weights
            accuracys.append(accuracy)
            print('Realização: {0}, Acurácia: {1}%'.format(i, accuracy))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys),
                                                                              np.std(accuracys)))
        print('Matrix de confusão da realização ', rel)
        print(selected_matrix)
        # print_points(inputs, w, test_data)

    # versicolor4INPUTS()

    def virginica4INPUTS():
        max_acuracy = 0
        rel = 0
        selected_matrix = []
        accuracys = []
        test_data, w = [], []
        inputs = ['petal_length', 'petal_width', 'petal_length', 'petal_width']
        perceptron.prepare_data("Iris-virginica", inputs)
        for i in range(20):
            perceptron.train(perceptron.iris_data)
            perceptron.test()
            accuracy, matrix = perceptron.accuracy_and_matrix()
            if (accuracy > max_acuracy):
                max_acuracy = accuracy
                rel = i
                selected_matrix = matrix
                test_data = perceptron.test_data
                w = perceptron.weights
            accuracys.append(accuracy)
            print('Realização: {0}, Acurácia: {1}%'.format(i, accuracy))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys),
                                                                              np.std(accuracys)))
        print('Matrix de confusão da realização ', rel)
        print(selected_matrix)
        # print_points(inputs, w, test_data)

    # virginica4INPUTS()

    def run_adaline_2d():
        data_set, weigths = [], []
        best_realization, min_mse = 0, 100
        adaline.start_artificial_dataset_2d(a=2, b=3)
        for i in range(20):
            adaline.train(adaline.data_set)
            adaline.test()
            mse = np.mean(adaline.realization_errors)
            rmse = np.sqrt(mse)
            if (mse < min_mse):
                best_realization, min_mse, data_set, weigths = i + 1, mse, adaline.data_set, adaline.weights
            print('Realização: ', i + 1, ', MSE: ', mse, ', RMSE: ', rmse)
        print('Melhor realizacao: ', best_realization, ', MSE: ', min_mse)
        adaline.plot(data_set, weigths)

    # run_adaline_2d()

    def run_adaline_3d():
        data_set, weigths = [], []
        best_realization, min_mse = 0, 100
        adaline.start_artificial_dataset_3d(a=3, b=2, c=1)
        for i in range(20):
            adaline.train_3d(adaline.data_set)
            adaline.test_3d()
            mse = np.mean(adaline.realization_errors)
            rmse = np.sqrt(mse)
            if (mse < min_mse):
                best_realization, min_mse, data_set, weigths = i + 1, mse, adaline.data_set, adaline.weights
            print('Realização: ', i + 1, ', MSE: ', mse, ', RMSE: ', rmse)
        print('Melhor realizacao: ', best_realization, ', MSE: ', min_mse)
        adaline.plot_3d(data_set, weigths)

    # run_adaline_3d()

    def run_single_layer_perceptron():
        accuracys = []
        inputs = [2, 3]
        single_layer_perceptron.prepare_data(1)
        for i in range(20):
            single_layer_perceptron.divide_data(single_layer_perceptron.data_set)
            single_layer_perceptron.train(inputs)
            accuracys.append(single_layer_perceptron.test(inputs))
            print('Realização: {0}, Acurácia: {1}%'.format(i + 1, accuracys[i]))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys),
                                                                              np.std(accuracys)))
        single_layer_perceptron.plot_decision_surface(inputs)

    run_single_layer_perceptron()

    def run_slp_artificial_dataset():
        accuracys = []
        inputs = [0, 1]
        single_layer_perceptron.generate_artificial_data(1)
        for i in range(20):
            single_layer_perceptron.divide_data(single_layer_perceptron.artiticial_data)
            single_layer_perceptron.train(inputs)
            accuracys.append(single_layer_perceptron.test(inputs))
            print('Realização: {0}, Acurácia: {1}%'.format(i + 1, accuracys[i]))

        print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(_calc_accuracy(accuracys),
                                                                              np.std(accuracys)))
        single_layer_perceptron.plot_decision_surface(inputs)

    # run_slp_artificial_dataset()


if __name__ == '__main__':
    main()
