import pandas as pd
import numpy as np
import copy 
import pickle
from .gramchd.gmd import sort_adjacencies
from graspologic.match import graph_match

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def get_w_threshold(dataframe, thrshld=None):
    '''
    Get connections above given threshold
    '''

    if thrshld==None:
        thrshld = 0
    
    df_thres = dataframe[dataframe['weight'] > thrshld]
    # df_thres = make_edges(df_thres)
    return df_thres


def relabel(input):

    '''
    getting rid of the connectivity suffix for all types
    '''
    copied = copy.deepcopy(input)

    c=0
    for i in range(len(copied)):
        if type(copied[i]) not in [float, type(None)]:
            try: 
                if copied[i][-2] == "_":
                    c +=1
                    # print(copied[i])
                    copied[i] = copied[i][:-2]
                    # print(copied[i])
            except:
                # print(copied[i])
                continue
    return copied, c


def get_neighbourhood(Ids, df):
    '''
    Will include NaN values
    '''
    source, sink = df.columns[:2]


    pre_post = df[df[sink].isin(Ids) | df[source].isin(Ids)].sort_values(by='weight', ascending=False).copy(True)

    # meta_dict = meta_df.set_index('id').to_dict()['type']
    # meta_dict[Id] = 'TARGET' + '_' + str(meta_dict[Id])
    # pre_post['pre_cat'] = pre_post[source].apply(lambda x: meta_dict[x]) # this will include NANS. 
    # pre_post['post_cat'] = pre_post[sink].apply(lambda x: meta_dict[x])

    return pre_post


def make_edges(df):
    _df = df.copy(True)

    _df['edges'] = _df.apply(lambda row: (row.bodyId_pre, row.bodyId_post), \
                             axis=1)
    return _df

def edgelist_to_adj(edgelist, indices):
    '''
    turn an edgelist into an adjacency list
    '''
    adj = [[] for i in indices]
    for i in edgelist:
        adj[indices[i[0]]].append(i[1])
        adj[indices[i[1]]].append(i[0])
    return adj

def get_weighted_adj_from_edgelist(edgelist):

    nodelist = set(np.concatenate(edgelist).tolist())
    indices = dict(zip(nodelist, range(len(nodelist))))
    adjlist = edgelist_to_adj(edgelist, indices) # create adjacency list from edgelist

    return indices, adjlist


def get_mapping_df(structure_1, structure_2, colnames, full_inds):
    '''Params:
    two dfs of ordered structure, 
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


def entropy_calc(array):
    '''
    Gets the shannon entropy of a vector whose elements are probabilities. Ignores zeros.
    i.e. vector.sum() = 1

    Even with less than ideal machine accuracy, entropy contribution of v --> 0^+ approaches zero.  
    '''
    if len(array[array>0]) < 1: # if the vector has no contributions, then return np.nan
        return np.nan 

    v = array[array > 0]
    # print(v)
    entropy = -sum(v * np.log(v))
    
    return entropy

def sort_adjacencies(id_1, id_2, graphs):
    '''Sorts the adjacencies of the two id's graph matrices by their size (number of nodes). 
    '''
    srtd = sorted([graphs[k] for k in [id_1, id_2]], key=lambda x:x.shape[1])
    adj_1 = srtd[0] # smaller graph
    adj_2 = srtd[1] # larger graph
    output = {'adj_1':adj_1, 'adj_2':adj_2}
    return output


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

# def get_mapping_df(structure_1, structure_2, colnames, full_inds):
#     '''Params:
#     two dfs of ordered structure, 
#     the full indices for which structure 2 must be indexed, 
#     colnames for which the first and second element indexes the respective dfs, 

#     return the df mapping the node matching with structure 1 order as the base. 
#     '''

#     structure_1_copy = structure_1.reset_index().rename(columns={'index':colnames[0]}).copy(True)
#     structure_2_copy = structure_2.iloc[full_inds].reset_index().rename(columns={'index':colnames[1]}).copy(True)

#     g2_to_g1_node_mapping = pd.concat([structure_1_copy, structure_2_copy], axis=1)
#     g2_to_g1_node_mapping.replace({'':np.nan}, inplace=True)
#     g2_to_g1_node_mapping.dropna(how='all', inplace=True)
#     return g2_to_g1_node_mapping