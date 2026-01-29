import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from .nhood import get_io_nodes


# from graspologic.match import graph_match

# copied from contact_maps


def get_node_typed(node_types, edge, directed=False):
    '''Given node_types dict, and a specific edge, get the string of size two. 
    if undirected, then the string is sorted in alphabetical order to count combinations rather than permutations. 
    '''
    if not directed:
        # t = ''.join(sorted(f'{node_types[edge[0]]}{node_types[edge[1]]}'))
        t = '_'.join(sorted([node_types[edge[0]], node_types[edge[1]]])) # accepts multistring node types and won't scramble strings within a type label. 
    else:
        t=f'{node_types[edge[0]]}_{node_types[edge[1]]}'
    return t


def edgelist_to_adjmat(edgelist, N, directed=False):
    '''Turn an edgelist into an adjacency matrix. N is the number of nodes
    Valid for when the edgelist is given by (i, j) where i and j are integer indices of the matrix. 
    '''
    adj_mat = np.zeros((N, N))

    for i in edgelist:
        adj_mat[i] = 1
        if not directed: # make symmetrical matrix if undirected. 
            adj_mat[i[::-1]] = 1

    return adj_mat

# def edgelist_to_multigraph_adj_mat(edgelist, node_types, directed=False):
#     '''Given edgelist, and the corresponding node_labels, get the adjacency matrix where each
#     dimension is the edge type according to the node_label. 

#     Relies on edgelist being indices from 0,...,N-1 where N is number of nodes. 
#     '''
#     node_type_list = sorted(set(node_types.values()))
#     edge_types = sorted([i+j for i in node_type_list for j in node_type_list])

#     G_edgelists = {i:[] for i in edge_types}
#     for i in edgelist:
#         if not directed:
#             t = ''.join(sorted(f'{node_types[i[0]]}{node_types[i[1]]}'))
#         else:
#             t=f'{node_types[i[0]]}{node_types[i[1]]}'
#         G_edgelists[t].append(i)

#     G_adj_mats = {i:edgelist_to_adjmat(edgelist=G_edgelists[i], N=len(node_types), directed=directed) for i in edge_types}
#     G_ml_adj_mat = np.array([G_adj_mats[i] for i in edge_types])
#     return G_ml_adj_mat




def prep_vars(adj_df, node_types):
    '''
    Given an adjacency dataframe, prepare the matrix, the types of edges that exist, and the node types. 
    
    requires node_types to be a dictionary of index in adj matrix to a string value. 
    '''
    string = adj_df.index.values # node names 
    node_names_dict = dict(zip(range(len(adj_df)), string))
    adj_mat = adj_df.to_numpy()

    # node_types = dict(zip(range(len(adj_mat)), [i[0] for i in string])) ### specific to the contact maps...

    node_type_list = sorted(set(node_types.values()))
    edge_types = sorted([i+j for i in node_type_list for j in node_type_list]) # concatenates the string.
    return adj_mat, edge_types

def prep_multiedgelist(adj_mat, edge_types, node_types, directed=False):
    '''Given adjacency matrix, possible edge_types and node_types dict, get the edgelist split by the different edge types. '''
    
    G_edgelists = {i:[] for i in edge_types}
    for i in range(len(adj_mat)):
        start=0
        if not directed:
            start = i+1
        for j in range(start, len(adj_mat)):
            if adj_mat[i,j] >0.5:
                # edge_list.append((i,j))
                t = get_node_typed(node_types=node_types, edge=(i,j), directed=directed)
                G_edgelists[t].append((i,j))
    return G_edgelists

def single_2_multiadjmat(adj_mat, edge_types, node_types, directed=False):
    '''From an adjacency matrix, make a multidimensional adj matrix by a separate dimension for each
    edge type determined from the node_types labels. '''

    G_edgelists = prep_multiedgelist(adj_mat=adj_mat, edge_types=edge_types, node_types=node_types, directed=directed)
    G_adj_mats = {i:edgelist_to_adjmat(edgelist=G_edgelists[i], N=len(node_types), directed=directed) for i in edge_types}
    G_ml_adj_mat = np.array([G_adj_mats[i] for i in edge_types])
    return G_ml_adj_mat

# def adj_df_to_multiedgelist(adj_df, directed=False):
#     # string = adj_df.index.values # node names 
#     # node_names_dict = dict(zip(range(len(adj_df)), string))
#     # adj_mat = adj_df.to_numpy()

#     # node_types = dict(zip(range(len(adj_mat)), [i[0] for i in string]))

#     # node_type_list = sorted(set(node_types.values()))
#     # edge_types = sorted([i+j for i in node_type_list for j in node_type_list])
#     adj_mat, edge_types, node_types = prep_vars(adj_df=adj_df)

#     # G_edgelists = {i:[] for i in edge_types}
#     # for i in range(len(adj_mat)):
#     #     start=0
#     #     if not directed:
#     #         start = i+1
#     #     for j in range(start, len(adj_mat)):
#     #         if adj_mat[i,j] >0.5:
#     #             # edge_list.append((i,j))
#     #             t = get_node_typed(node_types=node_types, edge=(i,j), directed=directed)
#     #             G_edgelists[t].append((i,j))
#     G_edgelists = prep_multiedgelist(adj_mat=adj_mat, edge_types=edge_types, node_types=node_types, directed=directed)
#     return G_edgelists

def get_mutliadjmat(adj_df, node_types, directed=False):
    '''Gets the multilayer adjacency matrix from the adjacency dataframe'''
    adj_mat, edge_types = prep_vars(adj_df=adj_df, node_types=node_types)

    G_ml_adj_mat = single_2_multiadjmat(adj_mat=adj_mat, edge_types=edge_types,\
                                        node_types=node_types, directed=directed)
    # G_edgelists = prep_multiedgelist(adj_mat=adj_mat, edge_types=edge_types, node_types=node_types, directed=directed)

    # G_adj_mats = {i:edgelist_to_adjmat(edgelist=G_edgelists[i], N=len(node_types), directed=directed) for i in edge_types}
    # G_ml_adj_mat = np.array([G_adj_mats[i] for i in edge_types])
    return G_ml_adj_mat


def iob_multi_adj(id, df, nhood, edge_types, directed=False):
    '''Given a nx graph with the nodes labelled and the 
    dict of in/out/both labels, get the multi adjacency matrix'''

    gtype = {True: nx.DiGraph, False:nx.Graph}

    adj_mat = nx.adjacency_matrix(nhood).todense()
    node_names = dict(zip(range(len(nhood.nodes())), nhood.nodes()))

    # multiadj:
    name_to_type = get_io_nodes(n_id=id, neighbourhood_nodes=list(node_names.values()), df=df)
    node_types = {i:name_to_type[node_names[i]] for i in node_names.keys()}
    G_ml_adj_mat = single_2_multiadjmat(adj_mat=adj_mat, edge_types=edge_types, node_types=node_types, directed=directed )
    return G_ml_adj_mat
