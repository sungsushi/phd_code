
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
    
    adj_2_nsize = adj_2.shape[1]
    adj_1_nsize = adj_1.shape[1]

    all_inds = list(range(adj_2_nsize))
    not_matched = [i for i in all_inds if i not in perm_inds]
    full_inds = np.concatenate([perm_inds, not_matched]).astype(int)
    full_perm_inds = [full_inds, full_inds]

    cont_size = [adj_2_nsize, adj_2_nsize]
    if len(adj_1.shape)==3:
        third_dim = adj_1.shape[0] # if there are multiadjs
        full_perm_inds.insert(0, list(range(third_dim)))
        cont_size = [third_dim, adj_2_nsize, adj_2_nsize]

    # perform the permutation but keep the extra nodes in the matrix
    adj_2_transformed = adj_2[np.ix_(*full_perm_inds)]

    # transform the smaller adjacency matrix into a 
    # container of size len(adj_2) with the extra elements filled with zeros.
    adj_1_transformed = np.zeros(cont_size) # initialise larger container

    if len(adj_1.shape)==3:
        adj_1_transformed[:third_dim, :adj_1_nsize, :adj_1_nsize] = adj_1 
    else:
        adj_1_transformed[:adj_1_nsize, :adj_1_nsize] = adj_1 

    fnorm = np.linalg.norm(adj_2_transformed - adj_1_transformed)
    return fnorm

def sort_adjacencies(id_1, id_2, graphs):
    '''Sorts the adjacencies of the two id's graph matrices by their size (number of nodes). 
    '''
    srtd = sorted([graphs[k] for k in [id_1, id_2]], key=lambda x:x.shape[1])
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

    perm_slice = [perm_inds, perm_inds]
    if len(adj_1.shape)==3:
        third_dim = adj_1.shape[0] # if there are multiadjs
        perm_slice = [list(range(third_dim)), perm_inds, perm_inds]

    # print(len(perm_inds))
    # print(adj_1.shape)
    # print(adj_2.shape)
    # print(perm_slice)

    mtrx_ij = adj_2[np.ix_(*perm_slice)]
    # print(mtrx_ij.shape)

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

    # for if there is no matching:
    # case 1: if there are no corresponding edges to permute. 
    try:
        _, perm_inds, _, _ = graph_match(adj_1, adj_2, **match_args)
    except:
        perm_inds = [i for i in range(adj_1.shape[1])]

    calc = distance_stats(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds, directed=directed)
    calc.update({'id_1':id_1, 'id_2':id_2})
    
    return calc

def get_matching_perm_inds(id_1, id_2, graphs, match_args):
    '''return only the permutation indices.'''
    output = sort_adjacencies(id_1=id_1, id_2=id_2, graphs=graphs)
    adj_1 = output['adj_1']# smaller graph
    adj_2 = output['adj_2'] # larger graph

    if match_args == None:
        match_args = {'rng':0, 'padding':'naive', 'transport':True}

    # for if there is no matching:
    # case 1: if there are no corresponding edges to permute. 
    try:
        _, perm_inds, _, _ = graph_match(adj_1, adj_2, **match_args)
    except:
        perm_inds = [i for i in range(adj_1.shape[1])]

    return perm_inds


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

        current_comp_ind, file_ind = self.check_prev_progress(proc_number)

        print(' ')
        print('Number of calculations/total:', len(id_split[current_comp_ind:]), '/', len(id_split))
        fpath = self.prefix + f"_{proc_number}_{file_ind}.parquet"
        df = pd.DataFrame(get_gmds(id_split=id_split[current_comp_ind:], graphs=self.graphs, fpath=fpath))
        print('Thread number finished:', proc_number, 'time taken:', time.time() - t)
        print(' ')
        df.to_parquet(fpath)
        return df
    
    def check_prev_progress(self, proc_number):
        id_split = np.array_split(self.trials, self.n_processes)[proc_number]
        prev_fpath = self.prefix + f"_{proc_number}_*.parquet"
        files = glob.glob(prev_fpath)

        # is_file = os.path.isfile(prev_fpath)
        new_ind = len(files)
        if new_ind==0:
            print(f'no current progress on {proc_number}. Start from scratch')
            return 0, new_ind
        
        prev_df = combine_vecs(files)
        progress = len(prev_df)
        prog_ids = prev_df.iloc[-1][['id_1', 'id_2']].to_numpy()
        check = all([i in id_split[progress-1] for i in prog_ids])
        if check:
            print('Test for progress:', check)
            print('progress up to', progress, prog_ids)
            print('comparisons left:', len(prev_df)-progress)
        return progress, new_ind # the index of the next comparison to do
        



