import pandas as pd
import numpy as np
# import copy 
import matplotlib.pyplot as plt

def label_point(x, y, val, ax, alpha=0.5, fontsize=10):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    # print(a)
    for i, point in a.iterrows():
        ax.text(point['x']+ ax.get_xlim()[1]*0.0, point['y'], str(point['val']), alpha=alpha, fontsize=fontsize)

