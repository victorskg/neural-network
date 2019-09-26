import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_and():
    data = create_points([0,0], '0')
    data = data.append(create_points([1,1], '1'), ignore_index=True)
    data = data.append(create_points([0,1], '0'), ignore_index=True)
    data = data.append(create_points([1,0], '0'),  ignore_index=True)
    
    return data

def get_or():
    data = create_points([0,0], '0')
    data = data.append(create_points([1,1], '1'), ignore_index=True)
    data = data.append(create_points([0,1], '1'), ignore_index=True)
    data = data.append(create_points([1,0], '1'),  ignore_index=True)
    
    return data

def create_points(source, _class):
    points = []
    for _ in range(50):
        coords = [source[i] + np.random.random() * 0.09 for i in range(2)]
        coords.append(_class)
        points.append(coords)          
    return pd.DataFrame(data=points, columns=['x', 'y', 'class'])