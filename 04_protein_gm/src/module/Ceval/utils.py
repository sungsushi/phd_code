from scipy.cluster.hierarchy import to_tree
import numpy as np


def generate_cluster_evolution(Z, targets, ind_to_id):
    '''
    Z is the linkage matrix

    targets is the list of nodes we want present in our cluster

    ind_to_id is the list of nodes by which to index to give the node id. 

    max_era is the number of evolutionary eras of the cluster we wish to extract. 
    '''


    rootnode, nodelist = to_tree(Z, rd=True)
    bools = [i.is_leaf() for i in nodelist]
    opp_bools = list(map(lambda x: not x, bools))
    leaves_are = np.array([i.get_id() for i in nodelist])[bools]
    not_leaves = np.array([i.get_id() for i in nodelist])[opp_bools]

    s_set= set(targets)
    lst = [i.pre_order(lambda x: ind_to_id[x.id]) for i in np.array(nodelist)[opp_bools]] # getting list of all clusters 
    clstr_sizes = [len(i) for i in lst] # sizes of clusters
    cluster_bool = np.array(clstr_sizes) >= len(s_set) # only consider clusters of at least size of the neuron numbers we want:

    rel_clusters = np.array(lst, dtype=object)[cluster_bool] # all clusters containing at least s_set length or more. 
    # condition=False
    i = 0
    inds = []
    j=0
    # condition = [False] * max_era # get n top clusters
    while True:
        isit = len(s_set - set(rel_clusters[i])) == 0
        if isit:
            # condition[j] = isit
            j+=1
            # inds.append(i)
            yield rel_clusters[i]
        i+=1
    i-=1
    # return rel_clusters[inds] 

