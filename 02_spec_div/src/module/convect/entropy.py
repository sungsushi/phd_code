import numpy as np 
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from scipy.cluster.hierarchy import fcluster, to_tree, dendrogram, linkage
from matplotlib.pyplot import cm

import copy 

def get_entropy(vector):
    '''
    Gets the shannon entropy of a vector whose elements are probabilities. Ignores zeros.
    i.e. vector.sum() = 1

    Even with less than ideal machine accuracy, entropy contribution of v --> 0^+ approaches zero.  
    '''
    if vector.isnull().values.all(): # if the vector has no contributions, then return np.nan
        return np.nan 

    v = vector[vector > 0].values
    entropy = -sum(v * np.log(v))
    
    return entropy
        

def get_io_entropy(dataframe, in_columns, out_columns):
    df = dataframe.copy(True)

    df['in_entropy'] = df.loc[:,in_columns].apply(get_entropy, axis=1)
    df['out_entropy'] = df.loc[:,out_columns].apply(get_entropy, axis=1)

    return df


