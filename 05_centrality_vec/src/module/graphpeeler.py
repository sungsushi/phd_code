'''
This is a probabilistic layer stratification for directed, weighted graphs. 
by Sung Soo Moon

Inspired by Schlegel et al., _eLife_ (2021)
'''

import numpy as np
import pandas as pd

def get_next_layer(df, start_ids, prepost='pre'):
    '''deterministic layer from a set of start ids'''
    plist = df.columns[:2]
    if prepost == 'post':
        plist=plist[::-1]
    visited = set(start_ids)
    while True:
        next_layer = set(df[df[plist[0]].isin(visited)][plist[1]])
        new_nodes = next_layer - visited
        visited |= new_nodes
        yield new_nodes


def get_next_layer_probabilistic(df, start_ids, prop_weight, prepost='pre', seed=0, correction=1):
    '''weighted layer from a set of start ids
    prop_weight is the column name for the proportional weight to base the random choice
    correction gives the proportion of edge weights needed to satisfy 100% visiting chance
    '''
    np.random.seed(seed) 
    
    plist = df.columns[:2] 
    if prepost == 'post': 
        plist = plist[::-1]
    visited = set(start_ids)

    while True: 
        try_to_visit = df[(df[plist[0]].isin(visited))&(~df[plist[1]].isin(visited))] # get rid of previously visited nodes.
        r = np.random.rand(len(try_to_visit))
        actually_visit = try_to_visit[try_to_visit[prop_weight]/correction > r]
        next_layer = set(actually_visit[plist[1]])
        new_nodes = next_layer - visited
        visited |= new_nodes
        yield new_nodes, try_to_visit


def layer_realisation(df, start_ids, prepost='pre', seed=0, prop_weight='in_prop_weight', correction=0.3, N=None):
    """
    Probabilistic model for generating layers of nodes from a graph.
    This function iteratively generates layers of nodes based on a probabilistic model. 
    Nodes are selected for the next layer based on probabilities defined in the input dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        A dataframe representing edges in the graph. It must contain a column specified by `prop_weight` 
        that holds the probabilities for selecting nodes.

    start_ids : list or set
        A collection of node IDs from which the layer generation starts.

    prepost : str, optional, default='pre'
        Specifies the direction of traversal in the graph. Can be 'pre' or 'post'.

    seed : int, optional, default=0
        Seed for the pseudorandom number generator to ensure reproducibility.

    prop_weight : str, optional, default='in_prop_weight'
        The column name in `df` where the probabilities for node selection are stored.

    correction : float, optional, default=0.3
        A threshold value. If the probability of a node exceeds this value, the node is 
        guaranteed to be included in the next layer.

    N : int, optional, default=None
        The termination condition. If no new nodes are added after `N` consecutive layers, 
        the process stops. If not provided, defaults to 2.
        
    Returns:
    --------
    dict
        A dictionary where keys are layer indices (starting from 0) and values are sets of node IDs 
        in each layer.
    Notes:
    ------
    - The process terminates if there are no nodes left to visit or if no new nodes are added 
      after `N` consecutive layers.
    - If `N` is not provided, a default value of 2 is used.
    """
    prob_layer_generator = get_next_layer_probabilistic(df=df, start_ids=start_ids, prepost=prepost, seed=seed, prop_weight=prop_weight, correction=correction)
    if N == None:
        print('N is not set. Default is 2.')
        N = 2
    start_set = set(start_ids)
    pl = 0
    pl_dict = {pl:start_set}
    layer_change_count = 0
    prev_unvisited = 0
    while True:
        next_layer, try_to_visit = next(prob_layer_generator)
        pl+=1
        pl_dict[pl] = next_layer
        layer_change_count += 1
        if abs(len(try_to_visit) - prev_unvisited) > 0:
            layer_change_count = 0 # if new nodes are taken in, then reset the count. 
        if len(try_to_visit)==0: # break if no nodes left...
            break
        if layer_change_count == N:
            break # if no new nodes are added after N layers, then stop.
        prev_unvisited = len(try_to_visit) 
    return pl_dict