
import pandas as pd
import numpy as np
from graspologic.match import graph_match
import networkx as nx

from .gramchd.gmd import distance_stats 
from .gramchd.nhood import get_io_top_NM
import time 


def get_ego_top_NM(Id, df, N, M):
    '''From Id, and given edges df, get the ego network of the top weighted M nodes and N edges between them. 
    
    The top nodes are calculated by summing in/out edge weights via Id. 
    '''
    source, sink = df.columns[:2]

    pre = df.query(f'{sink}==@Id')[[source, 'weight']]
    pre['weight'] = pre['weight']#/pre.weight.max()
    post = df.query(f'{source}==@Id')[[sink, 'weight']]
    post['weight'] = post['weight']#/post.weight.max()
    pre.set_index('pre').sort_values('weight')
    summed_prepost = post.join(pre.set_index('pre'), on='post', lsuffix='_post', rsuffix='_pre', how='outer').fillna(0)#.sort_values('weight_pre', ascending=False)
    summed_prepost.rename(columns={'post':'id'}, inplace=True)
    summed_prepost['total_weight'] = summed_prepost['weight_post'] + summed_prepost['weight_pre']
    pre_post_set = (summed_prepost.sort_values('total_weight', ascending=False).iloc[:M].id.tolist())

    # neigh_df = pre_post.iloc[:N] # edges - top nodes PARTNERED with the target. 
    all_ids = pre_post_set + [Id]
    n_edges = df[(df[source].isin(all_ids)) & (df[sink].isin(all_ids))]
    n_edges =  n_edges.sort_values(by='weight', ascending=False).iloc[:N] # top N edges in top M neurons. 
    return n_edges



def df_to_graph(df, directed=True, attr=False):
    ''' adds edges from dataframe df to a directed networkx graph '''
    if directed:
        g = nx.DiGraph()
    if not directed:
        g = nx.Graph()
    df_cols = ['pre','post']
    if attr:
        df_cols += ['attr']
        # print(df_cols)
    g.add_edges_from(df[df_cols].to_numpy())
    return g

def prepare_nhood_graphs(all_ids, df, edge_number, node_number, directed=True):
    '''Prepare the graph for all_ids given df, top edge_number edges in top node partners
    attr = True for attr addition to network.
    '''
    graphs = {}
    for i in all_ids:
        # print(i)
        edges = get_io_top_NM(Id=i, df=df, N=edge_number, M=node_number)
        uw_g = df_to_graph(edges, directed=directed,attr=False)
        w_g = df_to_graph(edges, directed=directed,attr=True)
        ego_edges = get_ego_top_NM(Id=i, df=df, N=edge_number, M=node_number)
        ew_g = df_to_graph(ego_edges, directed=directed,attr=True)
        graphs[i] = {}
        if len(uw_g)>0:
            graphs[i]['uw_adj_mat'] = nx.adjacency_matrix(uw_g).toarray()
        if len(w_g)>0:
            graphs[i]['w_adj_mat'] = nx.adjacency_matrix(w_g).toarray()
        if len(ew_g)>0:
            graphs[i]['ew_adj_mat'] = nx.adjacency_matrix(ew_g).toarray()
            graphs[i]['ew_g_nnames'] = list(ew_g.nodes())
        
    return graphs

def prepare_ego_nhood_graphs(all_ids, df, edge_number, node_number, directed=True):
    '''Prepare the graph for all_ids given df, top edge_number edges in top node partners
    attr = True for attr addition to network. Just the ego weighted neighbourhood...
    '''
    graphs = {}
    for i in all_ids:
        # print(i)
        ego_edges = get_ego_top_NM(Id=i, df=df, N=edge_number, M=node_number)
        ew_g = df_to_graph(ego_edges, directed=directed,attr=True)
        graphs[i] = {}
        if len(ew_g)>0:
            graphs[i]['ew_adj_mat'] = nx.adjacency_matrix(ew_g).toarray()
            graphs[i]['ew_g_nnames'] = list(ew_g.nodes())
    return graphs


def perform_matching(id_A, id_B, graphs, match_args=None):
    '''Does one graph matching given two ids and dict of graph matrices 
    id_1, id_2 : two ids to be compared
    graphs : dictionary of ids to adjacency matrices.
    match_args : for customisable matching distance arguments. Default is most basic. 

    returns a dict of 
    fnorm : the frobenius norm of just the matched part of the two matrices. 
    graph_edit_distance : the approximate edit distance between adj 1 and 2. 
    euclidean_distance : the frobenius norm of the transformed matrices. 
    isexact : True if adj_1 or the matched part of adj_2 is an exact subgraph of the other. 

    '''

    # output = sort_adjacencies(id_1=id_1, id_2=id_2, graphs=graphs)

    srtd = sorted([(graphs[k], k) for k in [id_A, id_B]], key=lambda x:x[0].shape[1])
    adj_1 = srtd[0][0] # smaller graph
    adj_2 = srtd[1][0] # larger graph
    id_1 = srtd[0][1]
    id_2 = srtd[1][1]
    # output = {'adj_1':adj_1, 'adj_2':adj_2}


    # adj_1 = output['adj_1']# smaller graph
    # adj_2 = output['adj_2'] # larger graph

    if match_args == None:
        match_args = {'rng':0, 'padding':'naive', 'transport':True}

    # for if there is no matching:
    # case 1: if there are no corresponding edges to permute. 
    try:
        _, perm_inds, _, _ = graph_match(adj_1, adj_2, **match_args)
    except:
        print('No matching:', id_1, id_2, flush=True)
        perm_inds = [i for i in range(adj_1.shape[1])]
    
    return perm_inds, id_1, id_2


def matching_distance(id_1, id_2, adj_1, adj_2, perm_inds, directed):
    '''Given two adjacency matrices where adj_2 >= adj_1 in size, and permutation indices of 
    
    '''
    calc = distance_stats(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds, directed=directed)
    calc.update({'id_1':id_1, 'id_2':id_2})
    return calc


##### ego weighted network:

def perform_ewg_matching_distance(id_A, id_B, graphs):

    id_A_ew_adj_mat = graphs[id_A]['ew_adj_mat']
    id_B_ew_adj_mat = graphs[id_B]['ew_adj_mat']
    gs = {id_A:id_A_ew_adj_mat, id_B:id_B_ew_adj_mat}


    srtd = sorted([(gs[k], k) for k in [id_A, id_B]], key=lambda x:x[0].shape[1])
    # adj_1 = srtd[0][0] # smaller graph
    # adj_2 = srtd[1][0] # larger graph
    id_1 = srtd[0][1]
    id_2 = srtd[1][1]
    
    id_1_ew_nnames = graphs[id_1]['ew_g_nnames']
    id_2_ew_nnames = graphs[id_2]['ew_g_nnames']

    one_index = np.arange(len(id_1_ew_nnames))[id_1 == np.array(id_1_ew_nnames)][0]
    two_index = np.arange(len(id_2_ew_nnames))[id_2 == np.array(id_2_ew_nnames)][0]
    pmatch = np.array([[one_index, two_index]])  # partial match seeding.

    perm_inds, id_1, id_2 = perform_matching(id_A=id_1, id_B=id_2, graphs=gs, match_args={'rng':0, 'padding':'naive', 'transport':True, 'partial_match':pmatch})
    
    # wgs = {id_A:id_A_w_adj_mat, id_B:id_B_w_adj_mat}
    # srtd = sort_adjacencies(id_1=id_A, id_2=id_B, graphs=wgs)
    wadj_1 = graphs[id_1]['ew_adj_mat']# smaller weighted adj mat
    wadj_2 = graphs[id_2]['ew_adj_mat']# larger weighted adj mat
    calc = matching_distance(id_1=id_A, id_2=id_B, adj_1=wadj_1, adj_2=wadj_2, perm_inds=perm_inds, directed=True)
    return calc 


def get_ewg_gmds(id_split, graphs):
    '''wrapper for getting multiple geds print time progress every 100 calculations. '''
    # t0 = time.time() # time 

    calcs = []
    # m = 0
    # hrly = 0
    before_time = time.time()
    for j in range(len(id_split)):
        i = id_split[j]
        id_1 = i[0]
        id_2 = i[1]
        calc = perform_ewg_matching_distance(id_A=id_1, id_B=id_2, graphs=graphs)
        calcs.append(calc)
        # time_now = time.time() - before_time
    #     if j // 1000 != m:
    #         print(j, 'out of', len(id_split), flush=True) # keep a track
    #         print('time:', time.time()-t0)

    #         print(' ')
    #         m = j // 1000

    #     if (time_now/3600)//1 != hrly:
    #         df = pd.DataFrame(calcs)
    #         df.to_parquet(fpath)
    #         hrly = (time_now/3600)//1
    return calcs




class ewg_gmd_wrapper:
    def __init__(self, prefix, trials, n_processes, graphs=None, gget_args=None):
        self.prefix = prefix
        self.trials = trials
        self.n_processes = n_processes
        self.graphs = graphs
        self.gget_args = gget_args

    def only_pnumber_needed(self, proc_number):
        id_split = np.array_split(self.trials, self.n_processes)[proc_number]
        graphs = self.graphs
        if graphs==None:
            ids = set(np.hstack(id_split))
            graphs = prepare_ego_nhood_graphs(all_ids=ids, **self.gget_args)

        t = time.time()
        fpath = self.prefix + f"_{proc_number}.parquet"
        df = pd.DataFrame(get_ewg_gmds(id_split=id_split, graphs=graphs, fpath=fpath))
        print(' ')
        print('Number of calculations:', len(id_split))
        print('Thread number finished:', proc_number, 'time taken:', time.time() - t)
        print(' ')
        df.to_parquet(fpath)
        return df