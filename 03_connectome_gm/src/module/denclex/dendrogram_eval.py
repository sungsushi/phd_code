import pandas as pd
import scipy as sp
import numpy as np

from .dendrograms_plotting import get_adj_from_edgelist, generate_cluster_evolution
from .tree_network import tree_search, get_network_from_tree


def pairwise_tree_dists(ids, adjlist, ind_dict):
    '''Pairwise tree distance calculation of all by all in ids list'''

    dist_calculations = []

    for i in range(len(ids)): 
        A_id = ids[i]
        for j in range(i+1, len(ids)):
            B_id = ids[j]
            dist = tree_search(nodes=[A_id], targets=[B_id], adjlist=adjlist, indices=ind_dict)
            dist_calculations.append({'A_id':A_id, 'B_id':B_id, 'tree_dist':dist[0]})

    dist_calculations = pd.DataFrame(dist_calculations)
    return dist_calculations


def cluster_purity_test(targets, Z, ind_to_id):
    '''From a set of nodes in targets, gather information about the sizes of pure clusters within the dendrogram, Z. 
    '''
    visited = set([])

    separated_clusters = []
    for i in targets:
        if i in visited: # if the clusters are already determined, then don't need to do the same calculation twice.
            continue
        
        cl_gen = generate_cluster_evolution(Z=Z, targets=[i], ind_to_id=ind_to_id)

        prev_cluster= [i]
        while True:

            next_cluster=next(cl_gen)
            condition = len(set(next_cluster) - set(targets)) > 0 # if the new cluster has more members than the targets, it's not pure anymore

            if condition: # terminate as prev_cluster is purest
                separated_clusters.append(prev_cluster)
                visited |= set(prev_cluster) 
                break
            prev_cluster = next_cluster
        if len(prev_cluster) == len(targets):
            # print('All nodes clustered together')
            break
    return separated_clusters


def get_network_from_tree_lm(Z):
    '''
    Low memory version without ind_to_id needed to be specified...
    Get the network representation using linkage matrix. 
    The node names will refer to the indices of the (condensed) distance matrix used to calculate Z.  

    params:
    Z : linkage matrix 
    ind_to_id : list of index ids in order of vectors used to create Z
    
    OUTPUT: 
    edge list of the tree network created by the dendrogram. 
    '''

    # Get dendrogramic representation:
    dendrograms = sp.cluster.hierarchy.dendrogram(Z, labels=None, no_plot=True)
    rootnode, nodelist = sp.cluster.hierarchy.to_tree(Z, rd=True) 
    bools = [i.is_leaf() for i in nodelist]
    opp_bools = list(map(lambda x: not x, bools))
    not_leaves = np.array([i.get_id() for i in nodelist])[opp_bools]


    lst = [i.pre_order(lambda x: x.id) for i in nodelist]
    nodelist=[int(i) for i in dendrograms['ivl']]
    # node_dicts = dict(zip(dendrograms['leaves'], nodelist))
    lst = [[int(j) for j in i] for i in lst]
    list_array = np.array(lst, dtype=object)[not_leaves].tolist()

    next_node_id = len(nodelist)

    edgelist= []
    while len(list_array) != 0:
        for i in list_array:
            if len(i) == 2:
                edgelist.append((i[0], next_node_id))
                edgelist.append((i[1], next_node_id))
                list_array.remove(i)
                for j in list_array:
                    if i[0] in j and i[1] in j:
                        j.remove(i[1])
                        j.remove(i[0])
                        j.append(next_node_id)
                next_node_id += 1

    return edgelist


def get_clustering_stats(Z, ctype_to_id, ind_to_id):
    '''Given:
    Z: linkage matrix
    ctype_to_id: node type : list of ids dictionary
    
    return the df containing the:
    largest pure cluster as a fraction of the total number in the cluster
    mean pariwise tree distance
    proportion of all members in the cluster in the smallest cluster that contains all members in the node type. 
    
    NEW: Uses get_network_from_tree_2 which doesn't require ind_to_id... saves edge_list with integer indices
    NEW: saves the number of pure clusters (>1 membership), number of singletons, and the number of members in pure clusters. 
    
    '''
    
    numerical_id_dict = dict(zip(ind_to_id,range(len(ind_to_id)))) # dictionary to idnices to string id. 

    edge_list = get_network_from_tree_lm(Z=Z)

    ind_dict, adjlist = get_adj_from_edgelist(edge_list) # getting ind dict is redundant if edge list is with numerical int labels

    labels = list(ctype_to_id.keys())

    lps = [] # largest pure clusters
    mpd = [] # mean pariwise tree distances
    pam = [] # proportion of ids of all members in the smallest cluster. 
    non_single_labels = []
    lens = []
    n_p = []
    m_p = []
    s_p = []

    for i in labels: 
        ids_to_inspect = ctype_to_id[i]
        if len(ids_to_inspect) ==1:
            continue
        non_single_labels.append(i)

        numerical_ids = [numerical_id_dict[ids] for ids in ids_to_inspect] # convert to numerical indices as adjlist is in numerical indices
        tdist_df = pairwise_tree_dists(ids=numerical_ids, adjlist=adjlist, ind_dict=ind_dict)
        mpd.append(tdist_df.tree_dist.mean())

        cl_purity = cluster_purity_test(targets=ids_to_inspect, Z=Z, ind_to_id=ind_to_id)
        largest_pure = max([len(j) for j in cl_purity])/len(ids_to_inspect)
        lps.append(largest_pure) 

        n_singletons = sum([len(i)==1 for i in cl_purity])
        n_non_singletons = len(cl_purity) - n_singletons
        n_non_singletons_members = len(ids_to_inspect) - n_singletons
        # average_members_per_pc = n_non_singletons_members/n_non_singletons
        n_p.append(n_non_singletons)
        m_p.append(n_non_singletons_members)
        s_p.append(n_singletons)

        cl_object = generate_cluster_evolution(Z=Z, targets=ids_to_inspect, ind_to_id=ind_to_id)
        smallest_cl = next(cl_object)
        prop_all_members = len(ids_to_inspect)/len(smallest_cl)
        pam.append(prop_all_members)
        lens.append(len(ids_to_inspect))
    cl_stats_df = pd.DataFrame(data={'labels':non_single_labels, 'len':lens, 'lps':lps, 'mpd':mpd, 'pam':pam, \
                                     'n_p':n_p, 'm_p':m_p, 's_p':s_p})
    

    return cl_stats_df
