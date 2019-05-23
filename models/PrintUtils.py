import numpy as np
import matplotlib.pyplot as plt

def print_multi_weigths(inputs, weigths, datas):
    setosa, versicolor, virginica = [], [], []
    for data in datas:
        selected_inputs = [data[inputs[j]] for j in range(len(inputs))]  
        if (data[len(data)-1] == '100'): setosa.append(selected_inputs)
        elif (data[len(data)-1] == '010'): versicolor.append(selected_inputs)
        else: virginica.append(selected_inputs)

    x, y = np.array(setosa).T
    w, z = np.array(versicolor).T
    k, h = np.array(virginica).T
    plt.xlabel('a')
    plt.ylabel('b')
    plt.plot(x,y, 'bo', w, z, 'go', k, h, 'co', scalex=3, scaley=3)
    plt.show()