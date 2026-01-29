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

from module.gmd import gmd_wrapper, combine_vecs
# from module.nhood_vis import df_to_graph
# from module.nhood import get_io_top_NM
# from module.utils import get_neighbourhood, get_w_threshold

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
    epoch = time.time()
    th = 5
    node_number = 50
    edge_number = None
    manc_edges, manc_meta = prep_data()
    thresholded_edges = get_w_threshold(manc_edges, thrshld=th)

    # all_ids = sorted(manc_meta.id.tolist())[:]
    nproc=64
    

    gprefix=f'../graphs/VNC/vnc_M{node_number}'
    graphs = combine_pickels(f'{gprefix}_*.pkl')
    # graphs = load_object(f'{gprefix}_0.pkl')

    nexist_ids = sorted(list(graphs.keys()))
    print('number of graphs:', len(nexist_ids))
    trials = []
    for i in range(len(nexist_ids)):
        for j in range(i+1, len(nexist_ids)):
            trials.append((nexist_ids[i], nexist_ids[j]))
    print('number of calculations:', len(trials))
    print(' ')
    print(' ')
    print('graph prep time: ', time.time()-epoch)
    print(' ')
    print(' ')
    t1 = time.time()
    # nproc = 32
    pool=multiprocessing.Pool(processes=nproc)
    trials = trials[:]    
    prefix = f'../hpc_runs/vnc_gmatches/vnc_th{th}'
    # prefix = '../hpc_runs/vnc_gmatches/vnc_test'
    wrap_obj = gmd_wrapper(trials=trials, graphs=graphs, prefix=prefix, n_processes=nproc)

    pool.map(wrap_obj.only_pnumber_needed,list(range(nproc)))

    combined = combine_vecs(f'{prefix}_*.parquet')
    combined.to_parquet(f'{prefix}_combined.parquet')

    print(' ')
    print('time for calculations', time.time() - t1)
    print(' ')

if __name__ == '__main__':
    t0 = time.time()
    foo()
    print(' ')
    print(' ')
    print('total time taken:', time.time()-t0)

