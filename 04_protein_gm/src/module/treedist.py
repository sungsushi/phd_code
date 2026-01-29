
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

# adapted over from VNC_analysis/tree_distance_and_entropy/tree_pair.py


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
    inds = [indices[i] for i in prev] # not efficient - do everything in indices!!! HAve ADJ LIST IN INDICES
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
    # concated = vectors
    # Z = cluster.hierarchy.linkage(concated, method="average", metric='cosine', optimal_ordering=True)
    dendrograms = cluster.hierarchy.dendrogram(Z, labels=ind_to_id, get_leaves=True, no_plot=True);
    rootnode, nodelist = cluster.hierarchy.to_tree(Z, rd=True) 

    bools = [i.is_leaf() for i in nodelist]
    opp_bools = list(map(lambda x: not x, bools))
    # leaves_are = np.array([i.get_id() for i in nodelist])[bools]
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

    nodelist = set(np.concatenate(edgelist).tolist())
    indices = dict(zip(nodelist, range(len(nodelist))))
    adjlist = edgelist_to_adj(edgelist, indices) # create adjacency list from edgelist

    return indices, adjlist


def tree_search(nodes, targets, adjlist, indices):
    '''
    performs the pairwise search from each node in nodes for the closest distance to any of the targets. 

    returns dists where dists[i] is the closest distance of nodes[i] to any of the targets. 

    '''
    # nodes = A_ids # leaves only in 'A's
    # targets = B_ids
    dists = []

    for i in nodes:
        before = []
        prev = [i] # begin with 1 node
        count = 0
        while True:
            # print(prev)
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



def get_closest_tree_dist(A_ids, B_ids, vectors):
    '''
    Get the tree distance between two sets of nodes in a dendrogram.

    return
        dists : the array of closest distances from A_ids to any of B_ids. 
    '''


    Z = cluster.hierarchy.linkage(vectors, method="average", metric='cosine', optimal_ordering=True)
    ind_to_id = vectors.index.values 

    edgelist = get_network_from_tree(Z, ind_to_id)

    indices, adjlist = get_adj_from_edgelist(edgelist) 

    # nodelist = set(np.concatentate(edgelist).tolist())
    # indices = dict(zip(nodelist, range(len(nodelist))))
    # adjlist = edgelist_to_adj(edgelist, indices) # create adjacency list from edgelist


    # nodes = A_ids # leaves only in 'A's
    # targets = B_ids
    # dists = []

    # for i in nodes:
    #     before = []
    #     prev = [i] # begin with 1 node
    #     count = 0
    #     while True:
    #         # print(prev)
    #         next = get_next_step(adjlist, prev, indices) # nodes in the next step
    #         test = set(next)-set(before) # don't take a step backwards
    #         before=prev # update 
    #         prev=next 
    #         isit = [i in targets for i in test] # Boolean testing for connection to B_ids
    #         count+=1 # update count (dist is one greater than the current count)
    #         if sum(isit)>0: # if True 
    #             # print(i)
    #             # print('Shortest distance to B_id:', count) 
    #             dists.append(count)
    #             break

    dists = tree_search(nodes=A_ids, targets=B_ids, adjlist=adjlist, indices=indices)
    return dists


# def get_pairwise_tree_dist(A_ids, B_ids, vectors):
#     '''
#     Get the tree distance between two sets of nodes in a dendrogram. This gets distances from all A_ids to all B_ids... 
#     '''


#     Z = cluster.hierarchy.linkage(vectors, method="average", metric='cosine', optimal_ordering=True)
#     ind_to_id = vectors.index.values 

#     edgelist = get_network_from_tree(Z, ind_to_id)

#     indices, adjlist = get_adj_from_edgelist(edgelist) 

#     # nodelist = set(np.concatentate(edgelist).tolist())
#     # indices = dict(zip(nodelist, range(len(nodelist))))
#     # adjlist = edgelist_to_adj(edgelist, indices) # create adjacency list from edgelist


#     # nodes = A_ids # leaves only in 'A's
#     # targets = B_ids
#     # dists = []

#     # for i in nodes:
#     #     before = []
#     #     prev = [i] # begin with 1 node
#     #     count = 0
#     #     while True:
#     #         # print(prev)
#     #         next = get_next_step(adjlist, prev, indices) # nodes in the next step
#     #         test = set(next)-set(before) # don't take a step backwards
#     #         before=prev # update 
#     #         prev=next 
#     #         isit = [i in targets for i in test] # Boolean testing for connection to B_ids
#     #         count+=1 # update count (dist is one greater than the current count)
#     #         if sum(isit)>0: # if True 
#     #             # print(i)
#     #             # print('Shortest distance to B_id:', count) 
#     #             dists.append(count)
#     #             break

#     dists = tree_search(nodes=A_ids, targets=B_ids, adjlist=adjlist, indices=indices)
    
#     return dists


    


# def get_mean_dist(A_id, B_id, thresh_df, meta_df, all_vector, del_others=False):
#     '''
#     A_id, B_id are ids of two neurons to be compared. 

#     all_vector is the connection vector of all the neurons. 

#     thresh_df is the thresholded dataframe of edges (conventionally threshold=0).

#     '''

#     # get downstream neurons from both A & B
#     A_desc = get_downstream(A_id, thresh_df)
#     A_desc = A_desc.sort_values(by='weight', ascending=False)
#     B_desc = get_downstream(B_id, thresh_df)
#     B_desc = B_desc.sort_values(by='weight', ascending=False)

#     # local interneurons:
#     l_i_neurons = meta_df[meta_df['class'] == 'Local Interneuron'].bodyId.values

#     # get only LNs from downstream:
#     A_filtered = A_desc[A_desc['bodyId_post'].isin(l_i_neurons)].sort_values(by='weight', ascending=False)
#     B_filtered = B_desc[B_desc['bodyId_post'].isin(l_i_neurons)].sort_values(by='weight', ascending=False)

#     # get list of connection ids. 
#     Acon = A_filtered.bodyId_post.values
#     A_connections = list(dict.fromkeys(Acon))[:50]

#     Bcon = B_filtered.bodyId_post.values
#     B_connections = list(dict.fromkeys(Bcon))[:50]


#     # A:
#     A = all_vector.loc[all_vector.index.isin(A_connections)].copy(True)
#     del A['in_entropy']
#     del A['out_entropy']
#     if del_others:
#         del A['other_in']
#         del A['other_out']

#     new_ind = [f'A{i}' for i in A.index.values]
#     new_ind_dict = dict(zip(A.index.values, new_ind))

#     # change the side so that we try to cluster for the contralateral neuron:
#     def change_side(reg):
#         if reg == 'L':
#             return 'R'
#         elif reg == 'R':
#             return 'L'
#         else:
#             return reg
#     new_cols = [i[:2] + change_side(i[2]) + i[3:] for i in A.columns.values ]
#     new_cols_dict = dict(zip(A.columns.values, new_cols))

#     A = A.rename(index=new_ind_dict, columns=new_cols_dict)

#     # B:
#     if len(B_connections) == 0: # if no connections exist return -1
#         return -1

#     B = all_vector.loc[all_vector.index.isin(B_connections)].copy(True)
#     del B['in_entropy']
#     del B['out_entropy']
#     if del_others:
#         del B['other_in']
#         del B['other_out']

#     new_ind = [f'B{i}' for i in B.index.values]
#     new_ind_dict = dict(zip(B.index.values, new_ind))

#     B = B.rename(index=new_ind_dict)


#     # Get dendrogramic representation:
#     concated = pd.concat([A, B])
#     linkagematrix = cluster.hierarchy.linkage(concated, method="average", metric='cosine', optimal_ordering=True)
#     dendrograms = cluster.hierarchy.dendrogram(linkagematrix, labels=concated.index.values, get_leaves=True, no_plot=True);
#     rootnode, nodelist = cluster.hierarchy.to_tree(linkagematrix, rd=True) 

#     bools = [i.is_leaf() for i in nodelist]
#     opp_bools = list(map(lambda x: not x, bools))
#     leaves_are = np.array([i.get_id() for i in nodelist])[bools]
#     not_leaves = np.array([i.get_id() for i in nodelist])[opp_bools]


#     # The network representation:
#     lst = [i.pre_order(lambda x: x.id) for i in nodelist]
#     nodelist=[str(i) for i in dendrograms['ivl']]
#     node_dicts = dict(zip(dendrograms['leaves'], nodelist))
#     lst = [[node_dicts[j] for j in i] for i in lst]
#     list_array = np.array(lst, dtype=object)[not_leaves].tolist()

#     edgelist= []
#     while len(list_array) !=0:
#         for i in list_array:
#             if len(i) == 2:
#                 edgelist.append((i[0], i[0] + i[1]))
#                 edgelist.append((i[1], i[0] + i[1]))
#                 list_array.remove(i)
#                 for j in list_array:
#                     if i[0] in j and i[1] in j:
#                         j.remove(i[1])
#                         j.remove(i[0])
#                         j.append(i[0] + i[1])


#     # Perform min tree distance calculation:
#     G = nx.Graph()
#     G.add_edges_from(edgelist)
#     indices = dict(zip(G.nodes, range(len(G.nodes))))
#     adjlist = edgelist_to_adj(edgelist, indices) # create adjacency list from edgelist


#     nodes = A.index.values # leaves only in 'A's
#     targets = B.index.values
#     dists = []

#     for i in nodes:
#         before = []
#         prev = [i] # begin with 1 node
#         count = 0
#         while True:
#             # print(prev)
#             next = get_next_step(adjlist, prev, indices) # nodes in the next step
#             test = set(next)-set(before) # don't take a step backwards
#             before=prev # update 
#             prev=next 
#             isit = [i in targets for i in test] # Boolean testing for connection to B
#             count+=1 # update count (dist is one greater than the current count)
#             if sum(isit)>0: # if True 
#                 # print(i)
#                 # print('Distance to B:', count) 
#                 dists.append(count)
#                 break
#     return np.mean(dists)