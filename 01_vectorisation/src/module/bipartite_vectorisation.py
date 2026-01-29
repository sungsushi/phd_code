import numpy as np 
import pandas as pd
import os
import time
from collections import Counter
import itertools
import pickle

def bpt_counter(Id, df, pre_post):
    '''count the frequencies of label according to meta_df of the secondary set
         associated with the primary nodes in the bipartite edges df'''
    pre, post, pre_cat, post_cat = pre_post
    key_connections = df[df[pre] == Id].copy(True)  
    bpt_count = Counter(key_connections[post_cat].values) 
    return bpt_count


def bpt_list(Id, df, pre_post):
     '''return the list of labels according to meta_df of the secondary set
         associated with the primary nodes in the bipartite edges df'''
     pre, post, pre_cat, post_cat = pre_post
     key_connections = df[df[pre] == Id][post_cat].tolist() 
     return key_connections

def agg_list(list_dict, ids, ignore=[]):
    '''Count the aggregated list from the list_dict[i in ids]
    ignore is a list of elements to not count...
    '''
    to_concat = [list_dict.get(i, []) for i in ids] # if there are no labels then have an empty list
    merged = list(itertools.chain(*to_concat))
    count = Counter(x for x in merged if x not in ignore)
    return count


def agg_list_2(list_dict, ids):
    '''Count the aggregated list from the list_dict[i in ids]
    ignore is a list of elements to not count...
    '''
    to_concat = [list_dict.get(i, []) for i in ids] # if there are no labels then have an empty list
    merged = list(itertools.chain.from_iterable(to_concat))
    count = Counter(merged)
    return count


def get_ud_bpt_vectors(ids, edges, bpt_dict, keep=None):
    '''Get dataframe of upstream, downstream aggregated counts of nodes in edges by bpt_dict labels
    keep is a list of column names to keep in the vectorisation and normalisation. If keep==None, then use all labels.
    
    
    '''
    us_dicts = []
    ds_dicts = []

    pre, post = edges.columns[:2]

    for i in ids:
        us = edges.query(f'{post}==@i')['pre'].tolist()
        ds = edges.query(f'{pre}==@i')['post'].tolist()


        us_count = dict(agg_list_2(list_dict=bpt_dict, ids=us))
        us_dicts.append(us_count)

        ds_count = dict(agg_list_2(list_dict=bpt_dict, ids=ds))
        ds_dicts.append(ds_count)

        
    us_vect_df = pd.DataFrame(us_dicts, index=ids)
    if keep!=None:
        us_vect_df = us_vect_df.iloc[:,us_vect_df.columns.isin(keep)]

    us_vect_df = us_vect_df.div(us_vect_df.sum(axis=1), axis=0).add_suffix('_in')


    ds_vect_df = pd.DataFrame(ds_dicts, index=ids)
    if keep!=None:
        ds_vect_df = ds_vect_df.iloc[:,ds_vect_df.columns.isin(keep)] 

    ds_vect_df = ds_vect_df.div(ds_vect_df.sum(axis=1), axis=0).add_suffix('_out')

    ud_vect_df = pd.concat([us_vect_df, ds_vect_df], axis=1)

    return ud_vect_df



def all_ud_bpt_vectors(ids, edges, bpt_dict, keep=None, prefix=''):
    '''
    wrapper for saving and retrieving upstream downstream aggregated bipartite labelled graph.
    '''
    fpath = f"{prefix}_ud_bpt_vectors.parquet"
    if os.path.isfile(fpath):
        print(f"{prefix} vector parquet exists.")
        all_vectors = pd.read_parquet(fpath)
        return all_vectors
    else:
        t0 = time.time() # time 
        all_vectors = get_ud_bpt_vectors(ids=ids, edges=edges, bpt_dict=bpt_dict, keep=keep)
        print(f'time elapsed for {prefix} vectorisation:', t0-time.time())
        all_vectors.to_parquet(fpath)
        return all_vectors

def get_entropy(vector, delta=1e-8):
    '''
    Gets the shannon entropy of a vector whose elements are probabilities. Ignores zeros.
    i.e. vector.sum() = 1

    Even with less than ideal machine accuracy, entropy contribution of v --> 0^+ approaches zero.  
    '''
    if vector.isnull().values.all(): # if the vector has no contributions, then return np.nan
        return np.nan 

    v = vector[vector > delta].values
    entropy = -sum(v * np.log(v))
    
    return entropy
    

def get_io_entropy(dataframe, in_columns, out_columns):
    df = dataframe.copy(True)

    df['in_entropy'] = df.loc[:,in_columns].apply(get_entropy, axis=1)
    df['out_entropy'] = df.loc[:,out_columns].apply(get_entropy, axis=1)

    return df


def get_bpt_dict(ids, bpt_df, pre_post, fpath_prefix):
    '''Wrapper for bipartite collapsing in btp_df. Saves the dictionary as a pickle file.'''
    fpath= fpath_prefix + '.pickle'
    if os.path.isfile(fpath):
        with open(fpath, 'rb') as handle:
            d = pickle.load(handle)
        return d

    d = {}
    for i in ids:
        ith_list = bpt_list(Id=i, df=bpt_df, pre_post=pre_post)
        d[i] = ith_list

    with open(fpath, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def row_count_subset(csrarray, r_ids, c_ids, row_names, col_names):
    '''
    row_names and col_names specify the csrarray row and column identity. 

    This function slices the csrarray by its row id names ``r_ids`` and column id names, ``c_ids``, 
    and sums over the r_ids to return a dict of c_ids and their frequencies (only those that have a non-zero count).
    
    '''

    cids_bool = pd.Series(col_names).isin(c_ids).values # bool of which columns correspond to c_ids of interest
    rids_bool = pd.Series(row_names).isin(r_ids).values # bool of which rows correspond to the r_ids of interest
    rids_go_array = csrarray[rids_bool] # csrarray of the data corresponding to r_ids of interest (row slicing)
    rids_bool_summed = rids_go_array.sum(axis=0) # column frequencies: summed over the r_ids
    sliced_go_counts = rids_bool_summed[cids_bool] # frequency for each column id that are in c_ids list 
    sliced_col_names = col_names[cids_bool] # column names present that are non-single GO terms

    # construct the dictionary of c_ids and the count within r_ids, only keep c_ids with at least one r_ids associated to it. 
    ids_counter = dict(zip(sliced_col_names[sliced_go_counts>0], sliced_go_counts[sliced_go_counts>0])) 
    return ids_counter