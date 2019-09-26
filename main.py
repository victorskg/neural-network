import numpy as np
import models.DataUtils as ut
from models.Adaline import Adaline
from models.Perceptron import Perceptron
from models.SingleLayerPerceptron import SingleLayerPerceptron

def main():
    #run_perceptron_iris()
    #run_perceptron_and()
    run_perceptron_or()
    #run_adaline_3d()
    #run_single_layer_perceptron()
    #run_slp_artificial_dataset()

def run_perceptron_iris():
    inputs = [0, 1, 2, 3]
    perceptron = Perceptron(learn_rate=0.1, max_epochs=30, data_path="datasets/iris.data")
    qt_trainning = int(0.8 * len(perceptron.data_set))
    np.random.shuffle(perceptron.data_set)
    train_data, test_data = perceptron.data_set[:qt_trainning], perceptron.data_set[qt_trainning:]
    perceptron.train(train_data, inputs)
    perceptron.test(test_data, inputs)

def run_perceptron_and():
    inputs = [0, 1]
    perceptron = Perceptron(learn_rate=0.1, max_epochs=10, data_path="datasets/iris.data")
    data_set = ut.get_and().to_numpy()
    np.random.shuffle(data_set)
    qt_trainning = int(0.8 * len(data_set))
    train_data, test_data = data_set[:qt_trainning], data_set[qt_trainning:]
    perceptron.train(train_data, inputs)
    perceptron.test(test_data, inputs)

def run_perceptron_or():
    inputs = [0, 1]
    perceptron = Perceptron(learn_rate=0.1, max_epochs=10, data_path="datasets/iris.data")
    data_set = ut.get_or().to_numpy()
    np.random.shuffle(data_set)
    qt_trainning = int(0.8 * len(data_set))
    train_data, test_data = data_set[:qt_trainning], data_set[qt_trainning:]
    perceptron.train(train_data, inputs)
    perceptron.test(test_data, inputs)

def run_adaline_2d():
    adaline = Adaline(learn_rate=0.1, max_epochs=200)

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


def run_adaline_3d():
    adaline = Adaline(learn_rate=0.1, max_epochs=200)

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


def run_single_layer_perceptron():
    single_layer_perceptron = SingleLayerPerceptron(learn_rate=0.1, neuron_count=3, max_epochs=500,
                                                    data_path="datasets/iris.data")

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


def run_slp_artificial_dataset():
    single_layer_perceptron = SingleLayerPerceptron(learn_rate=0.1, neuron_count=3, max_epochs=500,
                                                    data_path="datasets/iris.data")

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


def _calc_accuracy(array):
    return sum(array) / len(array)


if __name__ == '__main__':
    main()
