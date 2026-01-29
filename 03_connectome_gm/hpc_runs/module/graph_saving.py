import os
import time
import glob

import copy 
import networkx as nx
import numpy as np 
import pandas as pd
import multiprocessing


from module.gramchd.gmd import gmd_wrapper, combine_vecs
from module.gramchd.nhood_vis import df_to_graph
from module.gramchd.nhood import get_io_top_NM
from module.utils import get_neighbourhood, get_w_threshold, save_object, load_object
import pickle

def get_graphs(all_ids, df, edge_number, node_number, fpath):
    graphs = {}
    # nexist_ids = [] # list of ids that have non zero neighbourhoods
    for i in all_ids:
        # print(i)
        g = df_to_graph(get_io_top_NM(Id=i, df=df, N=edge_number, M=node_number))
        if len(g)>0:
            graphs[i] = nx.adjacency_matrix(g).todense().A
            # nexist_ids.append(i)
    save_object(graphs, fpath)
    
def combine_pickels(fpath_prefix):
    '''Dictionary'''
    files = glob.glob(fpath_prefix)
    combined = {}
    for i in files:
        j = load_object(i)
        combined.update(j)
    return combined



class gsaving_wrapper:
    def __init__(self, prefix, n_processes, Ids, node_number, edge_number, df, meta_df):
        self.ids = Ids
        self.df = df
        self.meta_df = meta_df
        self.prefix = prefix
        self.n_processes = n_processes
        self.node_number = node_number
        self.edge_number = edge_number

    def only_pnumber_needed(self, proc_number):
        id_split = np.array_split(self.ids, self.n_processes)[proc_number]
        fpath = self.prefix + f"_{proc_number}.pkl"
        get_graphs(all_ids=id_split, df=self.df, edge_number=self.edge_number, \
                   node_number=self.node_number, fpath=fpath)
        # df = pd.DataFrame(get_gmds(id_split, self.graphs))
        # df.to_parquet(fpath)
