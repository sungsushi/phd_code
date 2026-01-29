import numpy as np 
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import copy 

from .entropy import get_entropy, get_io_entropy 
from .vectorisation import get_all_vectorisation

def get_oneway_ent(all_vector, id_lst, df, keyed, in_columns, out_columns):
    '''
    Get the entropy of in/out of one way: upstream or downstream depending on keyed
        Parameters:
                all_vector : all annotation aggregated and normalised vectors. 
                id_lst : list of node ids to perform the entropy procedure
                df : dataframe of directed edge data where:
                        columns[0] is the start of the edge. 
                        columns[1] is the sink of the edge.
                        'weight' gives the weight of the edge. Set weights==1 for "unweighted" network. 
                keyed : specieis in/out by pre/post
                in_columns : the string labels for column_name + '_in'
                out_columns : the string labels for column_name + '_out'

        Returns:
                just_ents : concatenated specialisation-diversity vectors
    '''
    key_entropies = []
    for Id in id_lst:
        N_ids = df[df[keyed[0]] == Id]  # U/D ids to average individual vectors and entropies

        N_ids = N_ids.sort_values(by='weight', ascending=False) # sort by weights descending
        connections = N_ids[keyed[1]].drop_duplicates().values # pick non-duplicated top - no truncation

        if len(connections) == 0:
            continue
        c = all_vector.loc[all_vector.index.isin(connections)]

        meanc = c.mean(axis=0, skipna=True).reset_index() # explicitly skipna to avoid watering down probability (normalised properly)
        cols = meanc.loc[:,'index'].values # performs the mean of each column
        fields = meanc.loc[:, 0].values

        mean_df = pd.DataFrame(columns=cols)
        mean_df.loc[Id] = fields
        key_entropies.append(mean_df)

    all_keyed_io_entropies = pd.concat(key_entropies) 
    all_keyed_io_entropies['in_ent_of_avg'] = all_keyed_io_entropies.loc[:,in_columns].apply(get_entropy, axis=1)
    all_keyed_io_entropies['out_ent_of_avg'] = all_keyed_io_entropies.loc[:,out_columns].apply(get_entropy, axis=1)

    just_ents = all_keyed_io_entropies.iloc[:, -4:].copy(True) 
    just_ents['out_diff'] = just_ents['out_ent_of_avg'] - just_ents['out_entropy']
    just_ents['in_diff'] = just_ents['in_ent_of_avg'] - just_ents['in_entropy']
    return just_ents


def get_updown_entropy(df, meta_df, all_vector, id_lst=None, categories=None):
    '''get upstream and downstream specialisation-diversity:
    '''
    if id_lst is None:
        id_lst = all_vector.index.values
    source, sink = df.columns[:2]

    updown = {'D': [source, sink], 'U': [sink, source]}

    if categories is None:
        categories = [str(category) for category in meta_df.dropna().type.unique()] 
    rois = [i + j for i in categories for j in ['_in', '_out']] 
    in_columns = [i + '_in' for i in categories]
    out_columns = [i + '_out' for i in categories]


    ents = []
    for key in updown.keys():
        just_ent_key = get_oneway_ent(all_vector=all_vector, id_lst=id_lst, df=df, \
                                   keyed=updown[key], in_columns=in_columns, out_columns=out_columns)
        ents.append(just_ent_key)

    just_ents = ents[0].join(ents[1], how='outer', lsuffix='_ds', rsuffix='_us') # upstream and downstream entropies

    return just_ents

def get_ud_all_ent(df, meta_df, categories=None, fpath_prefix=''):

    '''
    Getting the UPSTREAM and DOWNSTREAM partners' in and out specialisation-diversities. 

    Saves as parquet
    '''

    fpath = fpath_prefix + "_ud_ents.parquet"
    if os.path.isfile(fpath):
        print(f"{fpath} upstream/downstream parquet csv exists.")
        
        all_entropy = pd.read_parquet(fpath)
        return all_entropy

    else:
        t0 = time.time() # time
        
        all_vector = get_all_vectorisation(df=df, meta_df=meta_df, fpath_prefix=fpath_prefix, categories=categories, entropy=True)
        just_ents = get_updown_entropy(df=df, meta_df=meta_df, all_vector=all_vector, categories=categories)

        print(f'time elapsed for specialisation-divesities:', time.time() - t0)

        just_ents.to_parquet(fpath)
        return just_ents



def ent_diff_hubs(all_entropies, ud, trunc):
    '''Get the entropic "hubs" 
    ud : {'us', 'ds'} 
    trunc = truncate top amount 
    '''
    tot_ent = all_entropies[f'out_diff_{ud}'] + all_entropies[f'in_diff_{ud}']
    tot_ent = tot_ent.sort_values(ascending=False)
    return tot_ent.iloc[:trunc]
