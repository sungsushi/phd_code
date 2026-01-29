import pandas as pd
import numpy as np
import re
import os

def celegans_data():
    ce_meta = prep_ce_meta()
    # lin_data = prep_lin_data()
    edge_df = prep_edge_df()
    cook_meta = prep_cook_meta()
    ce_meta = ce_meta.reset_index().join(cook_meta.add_prefix('Cook '), on='Neuron').set_index('Neuron')
    ce_meta['Cell Class'] = ce_meta['Cell Class'].astype(str)
    return edge_df, ce_meta


def prep_cook_meta():
    '''From Cook et al., Nature (2019) '''

    ce_meta = prep_ce_meta()

    sex_shared_label = pd.read_csv('../../data/celegans/Cell_lists_sexshared.csv', index_col=0)
    herm_label = pd.read_csv('../../data/celegans/Cell_lists_herm.csv', index_col=0, header=None)
    pharynx_label = pd.read_csv('../../data/celegans/pharynx_ctypes.csv', index_col=0, header=None)

    pharynx_label['cell category'] = "ph_" + pharynx_label[1]
    pharynx_label = pharynx_label.rename(columns={1:'cell type'})

    herm_label.columns = (sex_shared_label.columns[:3])

    sex_shared_df = sex_shared_label.loc[:, :'cell category']
    herm_specific_df = herm_label.loc[:, :'cell category']

    cook_meta = pd.concat([sex_shared_df, herm_specific_df, pharynx_label]) #.to_dict()
    cook_meta.index.name = 'Neuron'

    cook_meta = cook_meta.loc[ce_meta.index]
    return cook_meta

def prep_ce_meta():
    '''From Ripoll-Sanchez et al., Neuron (2023)'''
    ce_meta = pd.read_csv('../../data/celegans/072022_anatomical_class.csv', skiprows=2).set_index('Neuron')
    ce_meta.index = ce_meta.index.str.strip("'")
    return ce_meta

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return str(m.group()) if m else False

def remove_zeros(s):
    '''If string s ends in a number with placeholder zeros leading the number, we remove those zeros. '''
    tr_str = get_trailing_number(s)
    if tr_str:
        len_tr_str = len(tr_str)

        tr_int = str(int(tr_str))
        return s[:-len_tr_str] + tr_int
    return s

def index_nname_dict(x, nname_dict):
    try:
        return nname_dict[x]
    except:
        return x

def prep_edge_df():
    '''
    prepare Varsheney et al., (2011) (collected by from Witvliet et al., (2019)) edges with added zero placeholder for neuron labels.

    To be consistent with ce_meta data  
    '''
    ce_meta = prep_ce_meta()
    nname_dict = dict(zip(ce_meta.reset_index().Neuron.apply(lambda x: remove_zeros(x)), ce_meta.reset_index().Neuron))

    edge_df = pd.read_csv('../../data/celegans/white_1986_whole.csv', delim_whitespace=True)
    
    edge_df['pre'] = edge_df['pre'].apply(lambda x: index_nname_dict(x, nname_dict=nname_dict))
    edge_df['post'] = edge_df['post'].apply(lambda x: index_nname_dict(x, nname_dict=nname_dict))

    # filter out those without meta data:
    edge_df = edge_df[(edge_df['post'].isin(ce_meta.index)) \
                    & (edge_df['pre'].isin(ce_meta.index))]
    return edge_df

def syn_to_edge(edge_df, electrical=False):

    '''
    Given the white_1986_whole.csv, 
    Downloaded from Witvliet et al., nemanode (https://www.nemanode.org/)
    Data compiled from Varshney et al., (2011) and White et al., (1986)
    Turn into a weighted network. 

    Electrical=True treats electrical junctions as two edges of equal weight in both directions. 

    '''

    chem_edge_df = edge_df[edge_df['type']=='chemical'].copy(True)
    chem_edge_df = chem_edge_df[['pre', 'post', 'synapses']].rename(columns={'synapses':'weight'})

    if electrical:
        e_edge_df = elec_edge(edge_df=edge_df)
        e_c_edge_df = pd.concat([e_edge_df, chem_edge_df])

        e_c_edge_df = e_c_edge_df.groupby(['pre', 'post'], as_index=False).weight.sum()
        return e_c_edge_df
    
    chem_edge_df = chem_edge_df.groupby(['pre', 'post'], as_index=False).weight.sum()
    return chem_edge_df

def elec_edge(edge_df):

    '''
    Given the white_1986_whole.csv

    Turn into a weighted network. 

    Just outputs the electrical junctions as two edges of equal weight in both directions. 

    '''
    connections=[]
    e_edges = edge_df[edge_df['type']=='electrical']
    for i in range(len(e_edges)):
        connections.append({'pre':e_edges.iloc[i]['pre'], 'post':e_edges.iloc[i]['post'], 'weight':e_edges.iloc[i]['synapses']})
        connections.append({'post':e_edges.iloc[i]['pre'], 'pre':e_edges.iloc[i]['post'], 'weight':e_edges.iloc[i]['synapses']})

    e_edge_df = pd.DataFrame(connections)

    e_edge_df = e_edge_df.groupby(['pre', 'post'], as_index=False).weight.sum()
    return e_edge_df
    


