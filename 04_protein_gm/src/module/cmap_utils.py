import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from graspologic.match import graph_match


def binary_threshold(df, threshold):
    '''Thresholds the df into a binary matrix according to threshold'''
    _df = df.copy(True)
    _df[_df<=threshold]=1
    _df[_df>threshold]=0

    return _df


def prepare_adj_mat(fpath, threshold):
    '''prepares the adjacency matrix'''
    data = pd.read_csv(fpath, delimiter = "\t")
    data = data.iloc[:, :-1]
    data.set_index('ss_ele', inplace=True)
    data.replace(to_replace='-', value=np.inf, inplace=True)
    data = data.apply(pd.to_numeric)
    thresholded = binary_threshold(df=data, threshold=threshold)
    return thresholded

    