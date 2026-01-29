import numpy as np 
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from scipy.cluster.hierarchy import fcluster, to_tree, dendrogram, linkage
from matplotlib.pyplot import cm

import copy 



def get_io_vector(Id, dataframe, meta_df, rois, categories, labels): 
    '''
    GENERALISED version:

    :Id: Id of the node
    :dataframe: df of directed edge data, 
        columns[0] is the source of the edge. 
        columns[1] is the sink of the edge.
        'weight' gives the weight of the edge. Set weights==1 for "unweighted" network. 
    :meta_df: df of Id to category type mapping. 
    :rois: the strings for in/out catagories of interest. 
    :categories: the strings for the unique categories in meta_df
    :labels: the string labels for the source and sink in the dataframe. 
    

    :return: two vectors of weighted proportion to each catagory in/out of the Id node.

    NO OTHER_IN/OUT vector field - uninteresting if there are unlabelled data?

    '''
    source, sink = labels

    #Empty vector:
    vector = pd.DataFrame(columns=rois)
    vector.loc[Id] = None # initialise the vector
    # print(len(rois))

    connections = {'in': [sink, source, '_in', 'pre_cat', 'post_cat'], 
                   'out': [source, sink, '_out', 'post_cat', 'pre_cat']}

    for key in connections:
        

        key_connections = dataframe[dataframe[connections[key][0]] == Id].copy(True)

        # print(key_connections.roi.unique())
        key_nodes = meta_df[meta_df['id'].isin(key_connections[connections[key][1]].values)].copy(True)    
        key_nodes.set_index('id', inplace=True)

        if len(key_nodes) == 0: # if empty - continue
            continue #### if there's no "in" then continue to "out"

        key_vector = key_connections.groupby(connections[key][3]).agg({'weight':'sum'}).reset_index()  


        # We ignore extra vector columns that don't feature in our categories list. 
        # o_str = 'other' + connections[key][2]
        key_vector[connections[key][3]] = key_vector[connections[key][3]].apply(lambda x: str(x) + connections[key][2] if x in categories else None)
        # print(key_vector)

        key_vector = key_vector.dropna()
        key_vector['weight'] = key_vector['weight']/key_vector['weight'].sum()
        key_vector.columns = ['category', 'probability']

        # this aggregates all probabilities: not necessary if not wanting to take "other" fields
        # key_vector = key_vector['probability'].groupby(key_vector['category']).sum() 
        key_vector = key_vector.set_index('category').probability

        key_dict = key_vector.to_dict()
        # print(key_dict)
        vector.loc[Id].update(key_dict)
        # print(key_vector)

    return vector
    



def one_vectorisation(Id, df, meta_df, categories=None):
    """wrapper for vectorisation"""

    if not categories:
        categories = [str(category) for category in meta_df.dropna().type.unique()] # no need for ever looking at NaN vectorisation

    rois = [i + j for i in categories for j in ['_in', '_out']] 
    # print(len(rois))
    # target --> partner : "_out" 
    # target <-- partner : "_in" 

    source, sink = df.columns[:2]
    labels = [source, sink]

    _df = df.copy(True)
    meta_dict = meta_df.set_index('id').to_dict()['type']
    _df['pre_cat'] = _df[source].apply(lambda x: f"{meta_dict[x]}") # this will include NANS. 
    _df['post_cat'] = _df[sink].apply(lambda x: f"{meta_dict[x]}")

    vec = get_io_vector(Id=Id, dataframe=_df, meta_df=meta_df, rois=rois, \
                        categories=categories, labels=labels)

    return vec
    


# getting all vectorisation

def get_all_vectorisation(df, meta_df, fpath, Ids, categories=None ):
    '''
    Ids specify if we wish to take only a subset of nodes for vectorisation.  

    CHANGED 19/04/2023 : no option for Ids=None
    '''
    # fpath = f"./data/{prefix}_vec.parquet"
    if os.path.isfile(fpath):
        print(f"{fpath} vector parquet exists.")
        
        all_vector = pd.read_parquet(fpath)
        return all_vector
    else:
        t0 = time.time() # time 

        to_concat = []
        
        if not categories:
            categories = [str(category) for category in meta_df.dropna().type.unique()] # this doesn't include. 
        rois = [i + j for i in categories for j in ['_in', '_out']] 
        # target --> partner : "_out" 
        # target <-- partner : "_in" 
        
        source, sink = df.columns[:2]
        labels = [source, sink]

        # print(meta_dict)
        _df = df.copy(True) # doesnt make chanes to df
        meta_dict = meta_df.set_index('id').to_dict()['type']
        _df['pre_cat'] = _df[source].apply(lambda x: f"{meta_dict[x]}") # this will include NANS. 
        _df['post_cat'] = _df[sink].apply(lambda x: f"{meta_dict[x]}")

        # if not Ids: ################################ CHANGED: 30/03/2023 - have input ids to vectorise instead of all in metadata.
        #     Ids = set(_df[sink].values) | set(_df[source].values) 

        # for j in Ids:
        #     vec = get_io_vector(Id=j, dataframe=_df, meta_df=meta_df, rois=rois, \
        #                        categories=categories, labels=labels)
        #     to_concat.append(vec)
        m = 0
        for j in range(len(Ids)):
            vec = get_io_vector(Id=Ids[j], dataframe=_df, meta_df=meta_df, rois=rois, \
                    categories=categories, labels=labels)
            to_concat.append(vec)
            
            if j // 100 != m:
                print(j, 'out of', len(Ids)) # keep a track
                print('time:', time.time()-t0)
            m = j // 100

        all_vector = pd.concat(to_concat).fillna(1e-10)
        # all_vector = get_io_entropy(all_vector)

        print(f'time elapsed for {fpath} vectorisation:', t0-time.time())
        all_vector.to_parquet(fpath)
        return all_vector# hemi_vectors = get_all_vectorisation(df=hemi_edges, meta_df=hemi_meta, prefix='hemibrain')


# one_vec = one_vectorisation('357490931', hemi_edges, hemi_meta)
# one_vec




##### Not normalised:

def get_io_nn_vector(Id, dataframe, meta_df, rois, categories, labels): 
    '''
    GENERALISED version: 

    Does not normalise by the total in/out edge weights. 

    :Id: Id of the node
    :dataframe: df of directed edge data, 
        columns[0] is the source of the edge. 
        columns[1] is the sink of the edge.
        'weight' gives the weight of the edge. Set weights==1 for "unweighted" network. 
    :meta_df: df of Id to category type mapping. 
    :rois: the strings for in/out catagories of interest. 
    :categories: the strings for the unique categories in meta_df
    :labels: the string labels for the source and sink in the dataframe. 
    

    :return: two vectors of weighted proportion to each catagory in/out of the Id node.

    NO OTHER_IN/OUT vector field - uninteresting if there are unlabelled data?

    '''
    source, sink = labels

    #Empty vector:
    vector = pd.DataFrame(columns=rois)
    vector.loc[Id] = None # initialise the vector
    # print(len(rois))

    connections = {'in': [sink, source, '_in', 'pre_cat', 'post_cat'], 
                   'out': [source, sink, '_out', 'post_cat', 'pre_cat']}

    for key in connections:
        

        key_connections = dataframe[dataframe[connections[key][0]] == Id].copy(True)

        # print(key_connections.roi.unique())
        key_nodes = meta_df[meta_df['id'].isin(key_connections[connections[key][1]].values)].copy(True)    
        key_nodes.set_index('id', inplace=True)

        if len(key_nodes) == 0: # if empty - continue
            continue #### if there's no "in" then continue to "out"

        key_vector = key_connections.groupby(connections[key][3]).agg({'weight':'sum'}).reset_index()  


        # We ignore extra vector columns that don't feature in our categories list. 
        # o_str = 'other' + connections[key][2]
        key_vector[connections[key][3]] = key_vector[connections[key][3]].apply(lambda x: str(x) + connections[key][2] if x in categories else None)
        # print(key_vector)

        key_vector = key_vector.dropna()
        key_vector['weight'] = key_vector['weight'] # /key_vector['weight'].sum() # get rid of normalisation here....
        key_vector.columns = ['category', 'probability']

        # this aggregates all probabilities: not necessary if not wanting to take "other" fields
        key_vector = key_vector.set_index('category').probability

        key_dict = key_vector.to_dict()
        # print(key_dict)
        vector.loc[Id].update(key_dict)
        # print(key_vector)

    return vector


def one_nn_vectorisation(Id, df, meta_df, categories=None):
    """wrapper for vectorisation"""

    if not categories:
        categories = [str(category) for category in meta_df.dropna().type.unique()] # no need for ever looking at NaN vectorisation

    rois = [i + j for i in categories for j in ['_in', '_out']] 
    # print(len(rois))
    # target --> partner : "_out" 
    # target <-- partner : "_in" 

    source, sink = df.columns[:2]
    labels = [source, sink]

    _df = df.copy(True)
    meta_dict = meta_df.set_index('id').to_dict()['type']
    _df['pre_cat'] = _df[source].apply(lambda x: f"{meta_dict[x]}") # this will include NANS. 
    _df['post_cat'] = _df[sink].apply(lambda x: f"{meta_dict[x]}")

    vec = get_io_nn_vector(Id=Id, dataframe=_df, meta_df=meta_df, rois=rois, \
                        categories=categories, labels=labels)

    return vec


def get_all_nn_vectorisation(df, meta_df, categories=None, fpath=''):
    '''

    NOT Normalised version. 

    '''
    # fpath = f"./data/{prefix}_nn_vec.parquet"
    if os.path.isfile(fpath):
        print(f"{fpath} vector parquet exists.")
        
        all_vector = pd.read_parquet(fpath)
        return all_vector
    else:
        t0 = time.time() # time 

        to_concat = []
        
        if not categories:
            categories = [str(category) for category in meta_df.dropna().type.unique()] # this doesn't include. 
        rois = [i + j for i in categories for j in ['_in', '_out']] 
        # target --> partner : "_out" 
        # target <-- partner : "_in" 
        
        source, sink = df.columns[:2]
        labels = [source, sink]

        # print(meta_dict)
        _df = df.copy(True) # doesnt make chanes to df
        meta_dict = meta_df.set_index('id').to_dict()['type']
        _df['pre_cat'] = _df[source].apply(lambda x: f"{meta_dict[x]}") # this will include NANS. 
        _df['post_cat'] = _df[sink].apply(lambda x: f"{meta_dict[x]}")

        unique = set(_df[sink].values) | set(_df[source].values) 
        for j in unique:
            vec = get_io_nn_vector(Id=j, dataframe=_df, meta_df=meta_df, rois=rois, \
                               categories=categories, labels=labels)
            to_concat.append(vec)

        all_vector = pd.concat(to_concat).fillna(1e-10)
        # all_vector = get_io_entropy(all_vector)

        print(f'time elapsed for vectorisation:', t0-time.time())
        all_vector.to_parquet(fpath)
        return all_vector