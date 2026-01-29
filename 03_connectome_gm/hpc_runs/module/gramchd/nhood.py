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
    summed_prepost = post.join(pre.set_index('pre'), on='post', lsuffix='_post', rsuffix='_pre', how='outer').fillna(0)#.sort_values('weight_pre', ascending=False)
    summed_prepost.rename(columns={'post':'id'}, inplace=True)
    summed_prepost['total_weight'] = summed_prepost['weight_post'] + summed_prepost['weight_pre']
    pre_post_set = (summed_prepost.sort_values('total_weight', ascending=False).iloc[:M].id.tolist())

    # neigh_df = pre_post.iloc[:N] # edges - top nodes PARTNERED with the target. 

    n_edges = df[(df[source].isin(pre_post_set)) & (df[sink].isin(pre_post_set))]
    n_edges =  n_edges.sort_values(by='weight', ascending=False).iloc[:N] # top N edges in top M neurons. 
    return n_edges

def get_io_nodes(n_id, neighbourhood_nodes, df):
    '''
    Gets the set of both/in/out-nodes from the target to the neighbourhood.  
    ''' 
    pre, post = df.columns[:2]

    outedges = df[(df[pre]==n_id) & (df[post].isin(neighbourhood_nodes))] # there may be mulitple edges
    inedges = df[(df[post]==n_id) & (df[pre].isin(neighbourhood_nodes))] # there may be mulitple edges

    # weights:
    # in_e_weights = inedges.weight.values
    # out_e_weights = outedges.weight.values

    outnodes = set(outedges[post].values)
    innodes = set(inedges[pre].values)

    both = outnodes & innodes 

    outnodes = outnodes - both
    innodes = innodes - both

    iob_dict = dict(zip(outnodes, ['out' for _ in outnodes]))
    iob_dict.update(dict(zip(innodes, ['in' for _ in innodes])))
    iob_dict.update(dict(zip(both, ['both' for _ in both])))

    return iob_dict


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
