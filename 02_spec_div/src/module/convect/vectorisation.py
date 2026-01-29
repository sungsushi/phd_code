import numpy as np 
import pandas as pd
import os
import time
from .entropy import get_io_entropy

def get_io_vector(Id, dataframe, meta_df, rois, labels, io_cols): 
    '''
    Get in/out connectivity vectors for one given node.
    For a node Id present in an edgelist dataframe, we aggregate all in/out connectivity by the node annotations in meta_df.
        Parameters:
                Id : identifier of node in dataframe
                dataframe : dataframe of directed edge data where:
                        columns[0] is the start of the edge. 
                        columns[1] is the sink of the edge.
                        'weight' gives the weight of the edge. Set weights==1 for "unweighted" network. 
                meta_df : df of Id to category type mapping. 
                rois : arraylike of strings, annotations of interest to aggregate connectivity
                labels : the string labels for the source and sink in the dataframe (direction of edges)
                io_cols : io_cols['in'] is the in_column list, io_cols['out] is the out_column list.

        Returns:
                vector : concatenated vectors of weighted proportion to each catagory in/out of the Id node
    '''
    source, sink = labels

    #Empty vector:
    vector = pd.DataFrame(columns=rois)
    vector.loc[Id] = None # initialise the vector


    connections = {'in': [sink, source, '_in', 'pre_cat'], 
                    'out': [source, sink, '_out', 'post_cat']}

    for key in connections:
        key_connections = dataframe[dataframe[connections[key][0]] == Id].copy(True)

        cols = io_cols[key]

        key_nodes = meta_df[meta_df['id'].isin(key_connections[connections[key][1]].values)].copy(True)    
        key_nodes.set_index('id', inplace=True)

        if len(key_nodes) == 0: # if empty - intialise the container to be empty with np.nans 
            continue #### if there's no "in" then continue to "out"; if there's no "out" then stop. 

        key_vector = key_connections.groupby(connections[key][3]).agg({'weight':'sum'}).reset_index()  

        key_vector[connections[key][3]] = (key_vector[connections[key][3]] + connections[key][2])
        key_vector = key_vector[key_vector[connections[key][3]].isin(cols)] # only count the categories we wish to consider
        if len(key_vector)==0: # if none of the connections are the categories we are searching for, discard them.
            continue #### if there's no "in" then continue to "out"; if there's no "out" then stop. 

        key_vector['weight'] = key_vector['weight']/key_vector['weight'].sum()
        key_vector.columns = ['category', 'probability']
        key_vector = key_vector.set_index('category').probability

        # augment the other columns that are not present with zeros - stops it from being np.nans.
        key_dict = key_vector.to_dict()
        notin = [i for i in cols if i not in list(key_dict.keys())]
        key_dict.update({i:0 for i in notin})
        # add this row for the id to the vectors
        vector.loc[Id].update(key_dict)
    return vector
    


def one_vectorisation(Id, df, meta_df, categories=None):
    """wrapper for get_io_vector to get one node Id"""

    if not categories:
        categories = [str(category) for category in meta_df.dropna().type.unique()] # no need for ever consider NaN vectorisation

    rois = [i + j for i in categories for j in ['_in', '_out']] 
    # target --> partner : "_out" 
    # target <-- partner : "_in" 
    in_columns = [i + '_in' for i in categories]
    out_columns = [i + '_out' for i in categories]

    io_cols = {'in': in_columns, 'out':out_columns}

    source, sink = df.columns[:2]
    labels = [source, sink]

    _df = df.copy(True)
    meta_dict = meta_df.set_index('id').to_dict()['type']
    _df['pre_cat'] = _df[source].apply(lambda x: f"{meta_dict[x]}") 
    _df['post_cat'] = _df[sink].apply(lambda x: f"{meta_dict[x]}")

    vec = get_io_vector(Id=Id, dataframe=_df, meta_df=meta_df, rois=rois, \
                        labels=labels, io_cols=io_cols)

    return vec
    

def get_all_vectorisation(df, meta_df, Ids=None, categories=None, fpath_prefix='', entropy=False):
    '''
    Get in/out connectivity vectors for many nodes in Ids, and save them.
    For a node Id present in an edgelist df, we aggregate all in/out connectivity by the node annotations in meta_df.

    If a vector is already saved in the fpath_prefix + '_vec.parquet', then it will read and output instead of recalculating. 
            Parameters:
                    Ids : list; identifier of nodes in dataframe, if None then calculate vectors for all nodes present in df.
                    df : dataframe of directed edge data where:
                        columns[0] is the start of the edge. 
                        columns[1] is the sink of the edge.
                        'weight' gives the weight of the edge. Set weights==1 for "unweighted" network. 
                    meta_df : df of Id to category type mapping. 
                    categories : list of strings; specify a subset annotations in meta_df to aggregate connectivity. If None, then use all observed annotations in meta_df.
                    fpath_prefix : string; relative path to directory to save the vectors. 
                    entropy : if True, c

            Returns:
                    all_vector : the dataframe of all vectors, where the index is node ids, and the columns are in/out categories.
    '''    
    fpath= fpath_prefix + '_vec.parquet'
    if os.path.isfile(fpath):
        print(f"{fpath} vector parquet exists.")
        
        all_vector = pd.read_parquet(fpath)
        return all_vector
    else:
        t0 = time.time() # time 
        to_concat = []
        
        if not categories:
            categories = [str(category) for category in meta_df.dropna().type.unique()] 
        rois = [i + j for i in categories for j in ['_in', '_out']] 
        in_columns = [i + '_in' for i in categories]
        out_columns = [i + '_out' for i in categories]
        io_cols={'in':in_columns, 'out':out_columns}
        # target --> partner : "_out" 
        # target <-- partner : "_in" 
        
        source, sink = df.columns[:2]
        labels = [source, sink]

        _df = df.copy(True) # doesnt make changes to df
        meta_dict = meta_df.set_index('id').to_dict()['type']
        _df['pre_cat'] = _df[source].apply(lambda x: f"{meta_dict[x]}") 
        _df['post_cat'] = _df[sink].apply(lambda x: f"{meta_dict[x]}")

        if Ids is None: # have input ids to vectorise instead of all ids in metadata.
            Ids = set(_df[sink].values) | set(_df[source].values) 

        for j in Ids:
            vec = get_io_vector(Id=j, dataframe=_df, meta_df=meta_df, rois=rois, \
                                labels=labels, io_cols=io_cols)
            to_concat.append(vec)

        all_vector = pd.concat(to_concat) #.fillna(1e-10)
        if entropy:
            all_vector = get_io_entropy(all_vector, in_columns=in_columns, out_columns=out_columns)

        print(f'time elapsed for {fpath} vectorisation:', time.time()-t0)
        print("Entropy == ", entropy)
        all_vector.to_parquet(fpath)
        return all_vector


