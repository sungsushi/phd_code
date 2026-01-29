
## copied from 06_prepare_run.ipynb

import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import networkx as nx
import time 
from scipy.stats import pearsonr, kendalltau
from graspologic.match import graph_match
from .cmap_utils import prepare_adj_mat
from .gmd import distance_stats, sort_adjacencies


def get_orientation(p_id):
    '''output the orientation array'''
    fpath= f'./data/secondary_structure_contact_map/orientation/{p_id}.txt'
    orientation = pd.read_csv(fpath, delimiter = "\t")
    orientation = orientation.iloc[:, :-1]
    orientation.set_index('ss_ele', inplace=True)
    orientation.replace(to_replace='-', value=np.inf, inplace=True)
    orientation = orientation.apply(pd.to_numeric)
    return orientation

def get_orientation_multiadj(adj, orientation, types=False):
    ''' CORRECTED version: '''
    contact_orientation = (adj * orientation).fillna(0).astype(int)

    if not types: 
        types = list(set(orientation.replace([np.inf], 0).astype(int).to_numpy().flatten()) - {0.})
    # print(types)

    multiadj = np.zeros((len(types), len(contact_orientation), len(contact_orientation)))
    for t in range(len(types)):
        t_mat = (contact_orientation==types[t])*1
        multiadj[t, :len(contact_orientation), :len(contact_orientation)] = t_mat

    return multiadj


def chain_adj(size):
    '''Return an adjacency matrix where there are are edges between all consecutive elements.'''
    chain_adj = np.zeros((size, size))
    for i in range(size-1):
        chain_adj[i, i+1] = 1
        chain_adj[i+1, i] = 1
    return chain_adj

def dict_to_sim_mat(node_types_1, node_types_2, sim_val=1):
    '''Dicts are indexed by the indices of matrix 1 and 2 respectively 
    the similarity is given by matching the node types via the dictionary. 
    
    The dictionary is indexed by integers that correspond to indices of the adjacency matrix to match. 
    '''

    S = np.zeros((len(node_types_1), len(node_types_2)))

    for i in range(len(node_types_1)):
        for j in range(len(node_types_2)):
            if node_types_1[i] == node_types_2[j]:
                S[i,j] = sim_val
    return S


def get_node_types_2(adj_df):
    '''Slicing the first element in the index names of the adjacency dataframe as the node type
    returns a dictionary between the numerical index and the node type.'''
    string = adj_df.index.values # node names 

    node_types = dict(zip(range(len(adj_df)), [i[0] for i in string]))
    node_lengths = [int(i[1:]) for i in string]

    return node_types, node_lengths

def logistic(x):
    return 1/(1+np.exp(-x))

def length_similiarity(lengths, alpha, gamma):
    '''length similarity function'''
    delta = abs(lengths[0]-lengths[1])
    g_x = alpha * (1 - logistic(gamma*(delta-7.5))) # 7 or 8 quoted from runfeng
    return g_x
    # return alpha * ((2*min(lengths)/sum(lengths) )**gamma)


def dict_to_sim_mat_2(rawadj_1, rawadj_2, alpha=1, gamma=1, use_lengths=False):
    '''Dicts are indexed by the indices of matrix 1 and 2 respectively 
    the similarity is given by matching the node types via the the rawadj index names

    The lengths of the SSE elements are used to calculate similarity value

    alpha is the sim_value maximum. 
    '''
    node_types_1, node_lengths_1 = get_node_types_2(rawadj_1)
    node_types_2, node_lengths_2 = get_node_types_2(rawadj_2)

    S = np.zeros((len(node_types_1), len(node_types_2)))
    sim_val=alpha

    for i in range(len(node_types_1)):
        for j in range(len(node_types_2)):
            if node_types_1[i] == node_types_2[j]:
                lengths = [node_lengths_1[i], node_lengths_2[j]]
                if use_lengths:
                    sim_val = length_similiarity(lengths=lengths, alpha=alpha, gamma=gamma) 
                S[i,j] = sim_val
    return S


def get_cm_SS_order_graph(adj_df, col_name):
    '''Get the contact map adjacency matrix with a second layer as the SS chain order edge type from adjacency df 
    also returns the structure '''

    node_names = dict(zip(range(len(adj_df)), adj_df.index.values))
    adj_mat = adj_df.to_numpy()

    contact_multiadj = np.stack((adj_mat, chain_adj(len(adj_mat))), axis=0) ####### contact multiadj

    temp = nx.Graph()
    g = nx.from_numpy_array(adj_mat, create_using=temp)
    nx.set_node_attributes(g, node_names, name="CLASS")


    structure = pd.DataFrame({col_name:(node_names.values())}) 
    return contact_multiadj, g, structure


def get_node_types(adj_df):
    '''Slicing the first element in the index names of the adjacency dataframe as the node type
    returns a dictionary between the numerical index and the node type.'''
    string = adj_df.index.values # node names 

    node_types = dict(zip(range(len(adj_df)), [i[0] for i in string]))
    return node_types



def get_matching_full_inds(id_1, id_2, graphs, match_args=None):
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

    all_inds = list(range(adj_2.shape[1]))
    not_matched = [i for i in all_inds if i not in perm_inds]
    full_inds = np.concatenate([perm_inds, not_matched])


    return full_inds

def mapping_df(structure_1, structure_2, colnames, full_inds):
    '''Params:
    two dfs of ordered structure of the SS chains, 
    the full indices for which structure 2 must be indexed, 
    colnames for which the first and second element indexes the respective dfs, 

    return the df mapping the node matching with structure 1 order as the base. 
    '''

    structure_1_copy = structure_1.reset_index().rename(columns={'index':colnames[0]}).copy(True)
    structure_2_copy = structure_2.iloc[full_inds].reset_index().rename(columns={'index':colnames[1]}).copy(True)

    g2_to_g1_node_mapping = pd.concat([structure_1_copy, structure_2_copy], axis=1)
    g2_to_g1_node_mapping.replace({'':np.nan}, inplace=True)
    g2_to_g1_node_mapping.dropna(how='all', inplace=True)
    return g2_to_g1_node_mapping


def get_cm_SS_order_orientation_graph(adj_df, col_name, orientation):
    '''Get the contact map adjacency matrix with the orientation as
    the first three layers and last layer as the SS chain order edge type from adjacency df 
    also returns the structure '''

    node_names = dict(zip(range(len(adj_df)), adj_df.index.values))
    adj_mat = adj_df.to_numpy()

    temp = nx.Graph()
    g = nx.from_numpy_array(adj_mat, create_using=temp)
    nx.set_node_attributes(g, node_names, name="CLASS")

    structure = pd.DataFrame({col_name:(node_names.values())}) 

    o_multiadj = get_orientation_multiadj(adj=adj_df, orientation=orientation)
    contact_o_multiadj = np.concatenate((o_multiadj, np.array([chain_adj(o_multiadj.shape[1])])))

    return contact_o_multiadj, g, structure

def get_orientation_2(fpath):
    '''output the orientation array'''
    # fpath= f'./data/secondary_structure_contact_map/orientation/{p_id}.txt'
    orientation = pd.read_csv(fpath, delimiter = "\t")
    orientation = orientation.iloc[:, :-1]
    orientation.set_index('ss_ele', inplace=True)
    orientation.replace(to_replace='-', value=np.inf, inplace=True)
    orientation = orientation.apply(pd.to_numeric)
    return orientation


def find_order_discont(order):
    '''Given an array order, find the discontinuity '''
    difference = order[1:] - order[:-1] 
    
    # this finds the FINAL index in which the maximum discontinuity is found, \
    # if there are multiple discontinuities with the max value: 
    # max_diff = max((abs(v), i) for i, v in enumerate(difference)) 

    # this finds the FIRST index in which the maximum discontinuity is found, \
    # if there are multiple discontinuities with the max value: 
    max_diff = max((abs(v), -i) for i, v in enumerate(difference)) 


    ind = abs(max_diff[1]) # the biggest difference comes between ind and ind+1

    return ind + 1

def _excld(order, ind):
    '''If ind is first or last index, remove the trailing or leading element.'''
    if ind == 1:
        return order[1:]
    if ind == len(order)-1:
        return order[:-1]
    return np.array([False])


def discont_pearsonr(order):
    '''Given order of indices that is matching with list(range(len(order))), 
    find the pearson correlation. 
    
    If a discontinuity exists, then average the two pearsons either side of the 
    discontinuity. 
    
    Only suitable for one discontinuity. 
    
    '''
    discont_ind = find_order_discont(order=order) # finding the discontinuity. 
    total_pearson = pearsonr(x=list(range(len(order))), y=order)

    o = _excld(order, discont_ind)
    if o.any():  # unable to do pearson correltaion on one-size array
        trunc_pearson = pearsonr(x=list(range(len(o))), y=o)

        return trunc_pearson[0]
    # if discont_ind in [len(order)-1, 1]: # unable to do pearson correltaion on one-size array
    #     trunc_pearson = pearsonr(x=list(range(len(order[]))), y=order)

    #     return total_pearson[0]


    # two pearson correlation values either side of the discontinuity:
    first_pearson = pearsonr(x=list(range(discont_ind)), y=order[:discont_ind])
    second_pearson = pearsonr(x=list(range(discont_ind, len(order), 1)), y=order[discont_ind:])
    av_pearson = np.mean([first_pearson[0], second_pearson[0]])

    # total_pearson = pearsonr(x=list(range(len(order))), y=order)

    # if the average of the two pearson correlations at the discontinuity are higher than the total, return that:
    if abs(av_pearson) > abs(total_pearson[0]):
        # print('index', discont_ind)
        # print('First:', first_pearson)
        # print('Second:', second_pearson)
        return av_pearson
    
    return total_pearson[0]

def prepare_multi_cont_ori_adj_dict(ids, threshold=7.5, adj_fdir='./data/sec_network/sscm/', ori_fdir='./data/sec_network/orientation/'):
    '''Given ids, prepare a dict of the ids to their multilayer adjacency according to the 
    orientation edge type, and contact network as a layer. 
    '''

    output = {}
    for i in ids:

        # fpath = adj_fdir + i + '.txt'
        # adj = prepare_adj_mat(fpath=fpath, threshold=threshold)
        # o_fpath = ori_fdir + i + '.txt'
        # ori = get_orientation_2(fpath=o_fpath)

        # contact_o_multiadj, g, id_structure = get_cm_SS_order_orientation_graph(adj_df=adj, col_name='id_1_structure', orientation=ori)
        # node_types = get_node_types(adj_df=adj)

        output[i] = prepare_multi_cont_ori_adj(id=i, threshold=threshold, adj_fdir=adj_fdir, ori_fdir=ori_fdir)

    return output

def prepare_multi_cont_ori_adj(id, threshold=7.5, adj_fdir='./data/sec_network/sscm/', ori_fdir='./data/sec_network/orientation/'):
    '''Given an id, prepare a dict of the ids to their multilayer adjacency according to the 
    orientation edge type, and contact network as a layer. 
    '''

    fpath = adj_fdir + id + '.txt'
    adj = prepare_adj_mat(fpath=fpath, threshold=threshold)
    o_fpath = ori_fdir + id + '.txt'
    ori = get_orientation_2(fpath=o_fpath)

    contact_o_multiadj, g, id_structure = get_cm_SS_order_orientation_graph(adj_df=adj, col_name='id_1_structure', orientation=ori)
    node_types = get_node_types(adj_df=adj)

    output = {'node_types':node_types, 'contact_o_multiadj':contact_o_multiadj, 'g':g, 'id_structure':id_structure}

    return output


def dict_to_full_inds(id_A, id_B, id_A_dict, id_B_dict, S=10):
    '''
    Calculate the full indices of two graphs from running graph matching once.
    Get the order of the ids as well. 
    
    '''

    # sort so that we get smaller network first:
    srtd = sorted([id_A_dict | {'id':id_A}, id_B_dict | {'id':id_B}], key=lambda x:x['contact_o_multiadj'].shape[1])

    adj_1 = srtd[0]['contact_o_multiadj']# smaller graph
    node_types_1 = srtd[0]['node_types']
    id_1 = srtd[0]['id'] 
    # id_1_structure = srtd[0]['id_structure']

    adj_2 = srtd[1]['contact_o_multiadj'] # larger graph
    node_types_2 = srtd[1]['node_types']
    id_2 = srtd[1]['id'] 

    id_order =  [id_1, id_2]
    # id_2_structure = srtd[1]['id_structure']

    sim_mat = dict_to_sim_mat(node_types_1=node_types_1, node_types_2=node_types_2, sim_val=S)
    # calc = perform_matching_distance(id_1=id_1, id_2=id_2, graphs=gs, match_args={'rng':0, 'padding':'naive', 'transport':True, 'S':sim_mat})

    match_args={'rng':0, 'padding':'naive', 'transport':True, 'S':sim_mat}

    # for if there is no matching:
    # case 1: if there are no corresponding edges to permute. 
    try:
        _, perm_inds, _, _ = graph_match(adj_1, adj_2, **match_args)
    except:
        perm_inds = [i for i in range(adj_1.shape[1])]

    all_inds = list(range(adj_2.shape[1]))
    not_matched = [i for i in all_inds if i not in perm_inds]
    full_inds = np.concatenate([perm_inds, not_matched])

    return full_inds, id_order

def inds_to_stats(df, graph_dict):
    '''From a dataframe of ids to corresponding full indices, caluclate the graph stats. '''

    cdist_id_order = df[['id_1', 'id_2']].to_numpy()
    trials = df['id_order'].to_numpy()
    full_inds = df['full_inds'].tolist()

    calcs = []
    for i in range(len(trials)):
        id_1 = trials[i][0]
        id_1_graph = graph_dict[id_1]['contact_o_multiadj']
        id_1_size = id_1_graph.shape[1]

        id_2 = trials[i][1]
        id_2_graph = graph_dict[id_2]['contact_o_multiadj']

        perm_inds = [int(j) for j in full_inds[i][:id_1_size]]
        # try:
        calc = distance_stats(adj_1 = id_1_graph, adj_2=id_2_graph, perm_inds=perm_inds, directed=False)
        # except:
            # return id_1, id_2
        calc.update({'id_1':cdist_id_order[i][0], 'id_2':cdist_id_order[i][1]})

        calcs.append(calc)

    calcs_df = pd.DataFrame(calcs)
    return calcs_df
