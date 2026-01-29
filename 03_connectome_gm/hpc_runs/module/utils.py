import pandas as pd
import numpy as np
import copy 
import pickle

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


# def get_io_nodes(n_id, neighbourhood_nodes, df):
#     '''
#     Gets the set of both/in/out-nodes from the target to the neighbourhood.  
#     ''' 
#     pre, post = df.columns[:2]

#     outedges = df[(df[pre]==n_id) & (df[post].isin(neighbourhood_nodes))] # there may be mulitple edges
#     inedges = df[(df[post]==n_id) & (df[pre].isin(neighbourhood_nodes))] # there may be mulitple edges

#     # weights:
#     # in_e_weights = inedges.weight.values
#     # out_e_weights = outedges.weight.values

#     outnodes = set(outedges[post].values)
#     innodes = set(inedges[pre].values)

#     both = outnodes & innodes 

#     outnodes = outnodes - both
#     innodes = innodes - both

#     iob_dict = dict(zip(outnodes, ['out' for _ in outnodes]))
#     iob_dict.update(dict(zip(innodes, ['in' for _ in innodes])))
#     iob_dict.update(dict(zip(both, ['both' for _ in both])))

#     return iob_dict
