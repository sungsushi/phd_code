
import numpy as np
import pandas as pd
from graspologic.match import graph_match
import multiprocessing
import os
import time
import glob
import networkx as nx
import copy 


def matching_euclidean_distance(adj_1, adj_2, perm_inds):
    '''
    calculates the forbenius norm...
    adj_1 is smaller matrix, adj_2 is larger matrix. 
    perm_inds is a list of inds for adj_2 to be indexed into the matching to adj_1'''

    all_inds = list(range(len(adj_2)))
    not_matched = [i for i in all_inds if i not in perm_inds]
    full_inds = np.concatenate([perm_inds, not_matched]).astype(int)

    # perform the permutation but keep the extra nodes in the matrix
    adj_2_transformed = adj_2[np.ix_(full_inds, full_inds)]

    # transform the smaller adjacency matrix into a 
    # container of size len(adj_2) with the extra elements filled with zeros.
    adj_1_transformed = np.zeros((len(adj_2), len(adj_2))) # initialise larger container
    adj_1_transformed[:len(adj_1), :len(adj_1)] = adj_1 

    fnorm = np.linalg.norm(adj_2_transformed - adj_1_transformed)
    return fnorm

def sort_adjacencies(id_1, id_2, graphs):
    '''Sorts the adjacencies of the two id's graph matrices by their size (number of nodes). 
    '''
    srtd = sorted([graphs[k] for k in [id_1, id_2]], key=lambda x:len(x))
    adj_1 = srtd[0] # smaller graph
    adj_2 = srtd[1] # larger graph
    output = {'adj_1':adj_1, 'adj_2':adj_2}
    return output

def distance_stats(adj_1, adj_2, perm_inds, directed=True):
    '''Given the adjacency matrices and the permutation indices, get the distance statistics 
    if directed, then scale the values accordingly. 

    output:
    fnorm : the frobenius norm of just the matched part of the two matrices. 
    graph_edit_distance : the approximate edit distance between adj 1 and 2. 
    euclidean_distance : the frobenius norm of the transformed matrices. 
    isexact : True if adj_1 or the matched part of adj_2 is an exact subgraph of the other. 
    '''
    scale=1 
    if not directed:
        scale = 2 # don't overcount edge differences when undirected 

    mtrx_ij = adj_2[np.ix_(perm_inds, perm_inds)]

    fnorm = np.linalg.norm(adj_1 - mtrx_ij) # distance: the frobenius norm between the matched matrices

    test = ~((mtrx_ij - adj_1 <0.).any() * (mtrx_ij - adj_1 <0.).any())
    # if the matched graph_2 or graph_1 is a subgraph of the other, then ged calculation is guaranteed to be exact. 
    # print('Exact subgraph?', test)

    edge_d_b_ap = sum(abs(adj_1 - mtrx_ij)).sum() # edge edit distance between graph_1 and matched graph_2
    edge_d_a_ap = abs(sum(adj_2).sum() - sum(mtrx_ij).sum()) # edge number differnce between graph_2 and matched graph_2
    node_d_a_ap = abs(len(adj_2) - len(mtrx_ij)) # node number difference between graph_2 and matched graph_2
    graph_edit_distance = (edge_d_b_ap + edge_d_a_ap)/scale + node_d_a_ap # this is always at least an overestimate 
    euclidean_distance = matching_euclidean_distance(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds)

    calc = {'fnorm':fnorm, 'graph_edit_distance':int(graph_edit_distance), \
            'euclidean_distance':euclidean_distance, 'isexact':test}

    return calc


def perform_matching_distance(id_1, id_2, graphs, directed=True, match_args=None):
    '''Does one graph matching given two ids and dict of graph matrices 
    id_1, id_2 : two ids to be compared
    graphs : dictionary of ids to adjacency matrices.
    match_args : for customisable matching distance arguments. Default is most basic. 

    returns a dict of 
    fnorm : the frobenius norm of just the matched part of the two matrices. 
    graph_edit_distance : the approximate edit distance between adj 1 and 2. 
    euclidean_distance : the frobenius norm of the transformed matrices. 
    isexact : True if adj_1 or the matched part of adj_2 is an exact subgraph of the other. 

    '''

    output = sort_adjacencies(id_1=id_1, id_2=id_2, graphs=graphs)
    adj_1 = output['adj_1']# smaller graph
    adj_2 = output['adj_2'] # larger graph

    if match_args == None:
        match_args = {'rng':0, 'padding':'naive', 'transport':True}

    _, perm_inds, _, _ = graph_match(adj_1, adj_2, **match_args)

    calc = distance_stats(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds, directed=directed)
    calc.update({'id_1':id_1, 'id_2':id_2})
    
    return calc


# def perform_matching_distance(id_1, id_2, graphs):
#     '''Does one graph matching given two ids and dict of graphs and outputs:
#     fnorm between matched matrices 
#     test - whether the matched part of the graphs are exact subgraphs
#     graph edit distance - the approximate GED. 
#        '''
#     # id_1 = nexist_ids[i]
#     # id_2 = nexist_ids[j]
#     srtd = sorted([graphs[k] for k in [id_1, id_2]], key=lambda x:len(x))
#     adj_1 = srtd[0] # smaller graph
#     adj_2 = srtd[1] # larger graph

#     _, perm_inds, _, _ = graph_match(adj_1, adj_2, rng=0, padding='naive', transport=True)
#     mtrx_ij = adj_2[np.ix_(perm_inds, perm_inds)]

#     fnorm = np.linalg.norm(adj_1 - mtrx_ij) # distance: the frobenius norm between the matched matrices

#     test = ~((mtrx_ij - adj_1 <0.).any() * (mtrx_ij - adj_1 <0.).any())
#     # if the matched graph_2 or graph_1 is a subgraph of the other, then ged calculation is guaranteed to be exact. 
#     # print('Exact subgraph?', test)

#     edge_d_b_ap = sum(abs(adj_1 - mtrx_ij)).sum() # edge edit distance between graph_1 and matched graph_2
#     edge_d_a_ap = abs(sum(adj_2).sum() - sum(mtrx_ij).sum()) # edge number differnce between graph_2 and matched graph_2
#     node_d_a_ap = abs(len(adj_2) - len(mtrx_ij)) # node number difference between graph_2 and matched graph_2
#     graph_edit_distance = edge_d_b_ap + edge_d_a_ap + node_d_a_ap # this is always at least an overestimate 

#     return fnorm, graph_edit_distance, test

def get_gmds(id_split, graphs, fpath):
    '''wrapper for getting multiple geds print time progress every 100 calculations. '''
    t0 = time.time() # time 

    calcs = []
    m = 0
    hrly = 0
    before_time = time.time()
    for j in range(len(id_split)):
        i = id_split[j]
        id_1 = i[0]
        id_2 = i[1]
        calc = perform_matching_distance(id_1=id_1, id_2=id_2, graphs=graphs)
        calcs.append(calc)
        time_now = time.time() - before_time
        if j // 1000 != m:
            print(j, 'out of', len(id_split), flush=True) # keep a track
            print('time:', time.time()-t0)

            print(' ')
            m = j // 1000

        if (time_now/3600)//1 != hrly:
            df = pd.DataFrame(calcs)
            df.to_parquet(fpath)
            hrly = (time_now/3600)//1
    return calcs

def combine_vecs(fpath_prefix):
    files = glob.glob(fpath_prefix)
    
    opened = []
    for i in files:
        j = pd.read_parquet(i)
        opened.append(j)
    joined = pd.concat(opened, ignore_index=True)
    return joined



class gmd_wrapper:
    def __init__(self, prefix, trials, n_processes, graphs):
        # self.Ids = Ids
        # self.df = df
        # self.meta_df = meta_df
        self.prefix = prefix
        self.trials = trials
        self.n_processes = n_processes
        self.graphs = graphs

    def only_pnumber_needed(self, proc_number):
        id_split = np.array_split(self.trials, self.n_processes)[proc_number]
        t = time.time()
        fpath = self.prefix + f"_{proc_number}.parquet"
        df = pd.DataFrame(get_gmds(id_split=id_split, graphs=self.graphs, fpath=fpath))
        print(' ')
        print('Number of calculations:', len(id_split))
        print('Thread number finished:', proc_number, 'time taken:', time.time() - t)
        print(' ')
        df.to_parquet(fpath)
        return df
