import os
import time
import glob

import copy 
import networkx as nx
import numpy as np 
import pandas as pd
import multiprocessing


# from module.gmd import gmd_wrapper, combine_vecs
from module.nhood_vis import df_to_graph
from module.nhood import get_io_top_NM
from module.utils import get_neighbourhood, get_w_threshold, save_object, load_object
from module.graph_saving import gsaving_wrapper, combine_pickels

def prep_data():
    manc_edges = pd.read_csv('../data/vnc/20230529_data/manc_edges_20230529.csv', index_col=0)
    manc_edges = manc_edges.groupby(['bodyId_pre', 'bodyId_post'], as_index=False).weight.sum()
    manc_meta = pd.read_csv('../data/vnc/20230529_data/manc_meta_20230529.csv', index_col=0)
    
    manc_meta = manc_meta.loc[:, :].rename(columns={'bodyId':'id', 'class':'super_class', 'type':'cell_type', 'somaSide':'soma_side'})
    manc_meta['id'] = manc_meta['id'].astype(str)
    manc_edges.rename(columns={'bodyId_pre':'pre', 'bodyId_post':'post'}, inplace=True)
    manc_edges['pre'] = manc_edges['pre'].astype(str)
    manc_edges['post'] = manc_edges['post'].astype(str)
    return manc_edges, manc_meta


def foo():
    th = 2
    node_number = 50
    edge_number = None
    manc_edges, manc_meta = prep_data()
    thresholded_edges = get_w_threshold(manc_edges, thrshld=th)

    all_ids = sorted(manc_meta.id.tolist())[:]
    nproc=32

    pool=multiprocessing.Pool(processes=nproc)

    # th = 5
    # node_number = 50
    # edge_number = None
    prefix=f'../graphs/VNC/vnc_M{node_number}_th{th}'
    t_0 = time.time()

    wrap_obj = gsaving_wrapper(prefix=prefix, n_processes=nproc,\
                                Ids=all_ids, node_number=node_number,\
                                      edge_number=edge_number, \
                                        df=thresholded_edges, meta_df=manc_meta)
    
    pool.map(wrap_obj.only_pnumber_needed,list(range(nproc)))

    print(' ')
    print('time for calculations', time.time() - t_0)
    print(' ')

    combined = combine_pickels(f'{prefix}_*.pkl')
    print('number of combined graphs:', len(combined))
    save_object(combined, f'{prefix}_combined.pkl')

if __name__ == '__main__':
    t0 = time.time()
    foo()
    print(' ')
    print(' ')
    print('total time taken:', time.time()-t0)


