
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy import cluster
import copy

# adapted over from VNC_analysis/tree_distance_and_entropy/tree_pair.py & neuron_matching_hackathon
# From drosophila_nhood/src/denclex/tree_netowrk.py

def edgelist_to_adj(edgelist, indices):
    '''
    turn an edgelist into an adjacency list
    '''
    adj = [[] for i in indices]
    for i in edgelist:
        adj[indices[i[0]]].append(i[1])
        adj[indices[i[1]]].append(i[0])
    return adj


def get_next_step(adj_list, prev, indices):
    '''
    adj_list = adjacency list 
    prev = ids of nodes in the previous step 
    '''
    inds = [indices[i] for i in prev] 
    next_list = [adj_list[j] for j in inds]
    next_list = np.concatenate(next_list)
    next = list(set(next_list) - set(prev))
    return next


###### NEW:


def get_network_from_tree(Z, ind_to_id):
    '''
    params:
    Z : linkage matrix 
    ind_to_id : list of index ids in order of vectors used to create Z
    
    OUTPUT: 
    edge list of the tree network created by the dendrogram. 
    '''

    # Get dendrogramic representation:
    dendrograms = cluster.hierarchy.dendrogram(Z, labels=ind_to_id, get_leaves=True, no_plot=True);
    rootnode, nodelist = cluster.hierarchy.to_tree(Z, rd=True) 

    bools = [i.is_leaf() for i in nodelist]
    opp_bools = list(map(lambda x: not x, bools))
    not_leaves = np.array([i.get_id() for i in nodelist])[opp_bools]


    # The network representation:
    lst = [i.pre_order(lambda x: x.id) for i in nodelist]
    nodelist=[str(i) for i in dendrograms['ivl']]
    node_dicts = dict(zip(dendrograms['leaves'], nodelist))
    lst = [[node_dicts[j] for j in i] for i in lst]
    list_array = np.array(lst, dtype=object)[not_leaves].tolist()

    edgelist= []
    while len(list_array) !=0:
        for i in list_array:
            if len(i) == 2:
                edgelist.append((i[0], i[0] + i[1]))
                edgelist.append((i[1], i[0] + i[1]))
                list_array.remove(i)
                for j in list_array:
                    if i[0] in j and i[1] in j:
                        j.remove(i[1])
                        j.remove(i[0])
                        j.append(i[0] + i[1])
    
    return edgelist


def get_adj_from_edgelist(edgelist):
    '''From an edge list of node label strings, output the dictionary of indices and the adjacency list'''

    nodelist = set(np.concatenate(edgelist).tolist())
    indices = dict(zip(nodelist, range(len(nodelist))))
    adjlist = edgelist_to_adj(edgelist, indices) # create adjacency list from edgelist

    return indices, adjlist


def tree_search(nodes, targets, adjlist, indices):
    '''
    performs the pairwise search from each node in nodes for the closest distance to ANY of the targets. 

    returns dists where dists[i] is the closest distance of nodes[i] to any of the targets. 

    '''

    dists = []

    for i in nodes:
        before = []
        prev = [i] # begin with 1 node
        count = 0
        while True:
            next = get_next_step(adjlist, prev, indices) # nodes in the next step
            test = set(next)-set(before) # don't take a step backwards
            before=prev # update 
            prev=next 
            isit = [j in targets for j in test] # Boolean testing for connection to B_ids
            count+=1 # update count (dist is one greater than the current count)
            if sum(isit)>0: # if True 
                # print(i)
                # print('Shortest distance to B_id:', count) 
                dists.append(count)
                break

    return dists

def tree_type_search(nodes, targets, adjlist, indices, cl_dict):
    '''
    performs a search from each node in nodes for the closest distance to any of the targets. 

    returns:
        dists where dists[i] is the closest distance of nodes[i] to any of the targets
        clustered where clustered[i] is the categorisation according to cl_dict. 
        n_ids[i] is the ids of the nodes that gave the catagorisation in clustered[i]

    Each clustered[i] may be length >1 if there are multiple clusters with the same tree distance. 

    Outputs the node types of the closest matching tree distance according to cl_dict. 

    '''
    dists = []
    clustered = []
    node_ids = []

    for i in nodes:
        before = []
        prev = [i] # begin with 1 node
        count = 0
        while True:
            # print(prev)
            next = get_next_step(adjlist, prev, indices) # nodes in the next step
            test = list(set(next)-set(before)) # don't take a step backwards
            before=prev # update 
            prev=next 
            isit = [j in targets for j in test] # Boolean testing for connection to B_ids
            count+=1 # update count (dist is one greater than the current count)
            if sum(isit)>0: # if True 
                ids = np.array(test)[isit]
                node_ids.append(ids)
                types = set([cl_dict[k] for k in ids])
                # open to the possibility that there are multiple nodes with the same clustering. 
                clustered.append(types)
                dists.append(count)
                break

    return dists, clustered, node_ids



def get_closest_tree_dist(A_ids, B_ids, vectors, params=None):
    '''
    Get the tree distance between two sets of nodes in a dendrogram.

    return
        dists : the array of closest distances from A_ids to any of B_ids. 
    '''

    if params==None:
        params = {'method':"average", 'metric':'cosine', 'optimal_ordering':True}
    Z = cluster.hierarchy.linkage(vectors,**params)
    ind_to_id = vectors.index.values 

    edgelist = get_network_from_tree(Z, ind_to_id)

    indices, adjlist = get_adj_from_edgelist(edgelist) 
