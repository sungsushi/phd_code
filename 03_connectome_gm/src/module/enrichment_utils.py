import numpy as np
import scipy as sp
from scipy.cluster.hierarchy import to_tree
import pandas as pd

def get_clustersize_range(Z, ind_to_id, sizes):
    '''Get the clusters of at least size members. '''
    
    rootnode, nodelist = to_tree(Z, rd=True)
    bools = [i.is_leaf() for i in nodelist]
    opp_bools = list(map(lambda x: not x, bools))
    # leaves_are = np.array([i.get_id() for i in nodelist])[bools]
    # not_leaves = np.array([i.get_id() for i in nodelist])[opp_bools]

    # preorder == implicit that the clusters are all leaves! They're consistuent leaves...
    lst = [i.pre_order(lambda x: ind_to_id[x.id]) for i in np.array(nodelist)[opp_bools]] # getting list of all clusters 
    clstr_sizes = [len(i) for i in lst] # sizes of clusters
    cluster_bool = (np.array(clstr_sizes) >= sizes[0])*(np.array(clstr_sizes) <= sizes[1] )  # only consider clusters of at least size
    dists = [i.dist for i in np.array(nodelist)[opp_bools]] # getting distances of all clusters 

    rel_clusters = np.array(lst, dtype=object)[cluster_bool] # all clusters containing clstrsz number. 
    rel_dists = np.array(dists, dtype=object)[cluster_bool] # all cluster distances containing clstrsz number.|
    rel_cluster_tuples = [tuple(i) for i in rel_clusters]

    rel_df = pd.DataFrame({'cluster':rel_cluster_tuples, f'dists':rel_dists}).sort_values(by='dists', ascending=True)
    return rel_df

def hypergeometric_pmf(k, K, n, N):
    '''
    k : number of successes in sample
    K : number of total labelled successes in population
    n : number in sample
    N : total population size

    '''

    t1 = sp.special.comb(K,k, repetition=False)
    t2 = sp.special.comb(N-K,n-k, repetition=False)
    t3 = sp.special.comb(N, n, repetition=False)

    p = (t1 * t2 / t3)

    return p


def one_test(k, n, K, N):
    '''one-tailed hypothesis test'''
    ks = range(k, n+1) # assumes extremity for larger k values
    ps = 0.

    for kay in ks:
        p_val = hypergeometric_pmf(k=kay, K=K, n=n, N=N)
        ps += p_val
        # print(f'k = {kay},', 'p =', hypergeometric_pmf(k=kay, K=K, n=n, N=N))
        
    return ps