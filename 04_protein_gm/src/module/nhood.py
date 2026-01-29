import os
import time
import copy 


import numpy as np 
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx



def get_io_top_NM(Id, df, N, M):
    '''From Id, and given edges df, get the top weighted M nodes and N edges between them. 
    
    The top nodes are calculated by summing in/out edge weights via Id. 
    '''
    source, sink = df.columns[:2]

    pre = df.query(f'{sink}==@Id')[[source, 'weight']]
    pre['weight'] = pre['weight']#/pre.weight.max()
    post = df.query(f'{source}==@Id')[[sink, 'weight']]
    post['weight'] = post['weight']#/post.weight.max()
    pre.set_index('pre').sort_values('weight')
    summed_prepost = post.join(pre.set_index('pre'), on='post', lsuffix='_post', rsuffix='_pre').fillna(0)#.sort_values('weight_pre', ascending=False)
    summed_prepost.rename(columns={'post':'id'}, inplace=True)
    summed_prepost['total_weight'] = summed_prepost['weight_post'] + summed_prepost['weight_pre']
    pre_post_set = (summed_prepost.sort_values('total_weight', ascending=False).iloc[:M].id.tolist())

    # neigh_df = pre_post.iloc[:N] # edges - top nodes PARTNERED with the target. 

    n_edges = df[(df[source].isin(pre_post_set)) & (df[sink].isin(pre_post_set))]
    n_edges =  n_edges.sort_values(by='weight', ascending=False).iloc[:N] # top N edges in top M neurons. 
    return n_edges

# def get_io_top_NM(Id, df, N, M):
#     '''From Id, and given edges df, get the top weighted M nodes and N edges between them. 
    
#     The top nodes are calculated by summing in/out edge weights via Id. 
#     '''
#     source, sink = df.columns[:2]

#     pre = df.query(f'{sink}==@Id')[[source, 'weight']]
#     pre['weight'] = pre['weight']#/pre.weight.max()
#     post = df.query(f'{source}==@Id')[[sink, 'weight']]
#     post['weight'] = post['weight']#/post.weight.max()
#     pre.set_index('pre').sort_values('weight')
#     summed_prepost = post.join(pre.set_index('pre'), on='post', lsuffix='_post', rsuffix='_pre').fillna(0)#.sort_values('weight_pre', ascending=False)
#     summed_prepost.rename(columns={'post':'id'}, inplace=True)
#     summed_prepost['total_weight'] = summed_prepost['weight_post'] + summed_prepost['weight_pre']
#     pre_post_set = (summed_prepost.sort_values('total_weight', ascending=False).iloc[:M].id.tolist())

#     # neigh_df = pre_post.iloc[:N] # edges - top nodes PARTNERED with the target. 

#     n_edges = df[(df[source].isin(pre_post_set)) & (df[sink].isin(pre_post_set))]
#     n_edges =  n_edges.sort_values(by='weight', ascending=False).iloc[:N] # top N edges in top M neurons. 
#     return n_edges
