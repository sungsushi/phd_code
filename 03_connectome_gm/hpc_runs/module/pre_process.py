# import os
# import time
# import copy 
# from operator import itemgetter
# from collections import OrderedDict


import numpy as np 
import pandas as pd

# import plotly.express as px
# import plotly.graph_objects as go
# from matplotlib.pyplot import cm
# import matplotlib.pyplot as plt
# import seaborn as sns

# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import fcluster, to_tree, dendrogram, linkage

# from .vectorisation import one_vectorisation, get_all_vectorisation

# import nglscenes as ngl
def hb_type_to_ctype(x):
    if type(x.incomplete_cell_type)==str:
        return x.incomplete_cell_type 
    else:
        return x.hemibrain_type


def get_fw_data():
    fw_edges = pd.read_feather('./data/prepped_data/fw_edges.feather')
    fw_meta = pd.read_csv('./data/prepped_data/fw_meta.csv')
    fw_meta= fw_meta.loc[:,'row_id':].reset_index(drop=True)


    node_counts = fw_meta.value_counts("id")
    dup_nodes = fw_meta.query(
        "id.isin(@node_counts[@node_counts > 1].index)"
    ).sort_values("id")
    keep_rows = (
        dup_nodes.sort_values("hemibrain_type")
        .drop_duplicates("id", keep="first")
        .index
    )
    drop_rows = dup_nodes.index.difference(keep_rows)
    # fw_meta_copy = fw_meta.copy(True)
    fw_meta.drop(drop_rows, inplace=True)


    fw_meta['id'] = fw_meta['id'].astype(str)

    no_meta_fw = (set(fw_edges.post.unique()) | set(fw_edges.pre.unique())) - set(fw_meta.id.unique())
    fw_edges = fw_edges[~(fw_edges['post'].isin(no_meta_fw) | fw_edges['pre'].isin(no_meta_fw))]

    fw_meta.rename(columns={'side':'soma_side', 'cell_type':'incomplete_cell_type'}, inplace=True) # existing cell type label: incomplete

    fw_meta['cell_type'] = fw_meta.apply(lambda x: hb_type_to_ctype(x), axis=1) # new cell type labelling propagate HB labels. 


    return fw_meta, fw_edges



def get_mcns_data():
    mcns_edges = pd.read_feather('./data/prepped_data/mcns_edges.feather')
    mcns_meta = pd.read_csv('./data/prepped_data/mcns_meta.csv')

    mcns_meta['id'] = mcns_meta['id'].astype(str)

    no_meta_mcns = (set(mcns_edges.bodyId_post.unique()) | set(mcns_edges.bodyId_pre.unique())) - set(mcns_meta.id.unique())

    mcns_meta = mcns_meta.loc[:, 'id':].rename(columns={'class':'super_class', 'type':'cell_type', 'somaSide':'soma_side'})

    return mcns_meta, mcns_edges


def get_hb_data():
    hb_edges = pd.read_csv('./data/data_dumps/hemibrain/hemibrain_edges.csv')
    hb_meta = pd.read_csv('./data/data_dumps/hemibrain/hemibrain_meta.csv')

    hb_edges = hb_edges.groupby(['bodyId_pre', 'bodyId_post'], as_index=False).weight.sum()
    hb_edges = hb_edges.rename(columns={'bodyId_pre':'pre', 'bodyId_post':'post'})
    hb_edges['pre'] = hb_edges['pre'].astype(str)
    hb_edges['post'] = hb_edges['post'].astype(str)


    hb_meta = hb_meta.rename(columns={'bodyId':'id', 'type':'cell_type'})
    hb_meta['id'] = hb_meta['id'].astype(str)
    return hb_meta, hb_edges

# common_labels = (set(fw_meta.cell_type.dropna()) & set(mcns_meta.type.dropna()))



