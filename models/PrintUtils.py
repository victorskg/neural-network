import numpy as np
import matplotlib.pyplot as plt

def print_multi_weigths(inputs, weigths, datas):
    setosa, versicolor, virginica = [], [], []
    for data in datas:
        selected_inputs = [data[inputs[j]] for j in range(len(inputs))]  
        if (data[4] == '100'): setosa.append(selected_inputs)
        elif (data[4] == '010'): versicolor.append(selected_inputs)
        else: virginica.append(selected_inputs)
        
    a = np.linspace(0 ,10,100)
    for w in weigths:
        b = (-(w[0]/w[2]) / (w[0]/w[1]))*a + (-w[0] / w[2])
        plt.plot(a, b, '-r')

    x, y = np.array(setosa).T
    w, z = np.array(versicolor).T
    k, h = np.array(virginica).T
    plt.xlabel('a')
    plt.ylabel('b')
    plt.plot(x,y, 'bo', w, z, 'go', k, h, 'co')
    plt.show()