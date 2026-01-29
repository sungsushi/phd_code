
import numpy as np
from scipy.cluster.hierarchy import to_tree
import pandas as pd
from matplotlib.pyplot import cm
from scipy.cluster.hierarchy import to_tree, dendrogram
import matplotlib.pyplot as plt

from collections import Counter

from .tree_network import get_adj_from_edgelist, get_network_from_tree, tree_type_search



def get_clustersize(Z, clstrsz, ind_to_id):
    '''

    Search the Z matrix for clusters of size clstrsz 

    ind_to_id is the list of nodes by which to index to give the node id. 

    '''
    rootnode, nodelist = to_tree(Z, rd=True)
    bools = [i.is_leaf() for i in nodelist]
    opp_bools = list(map(lambda x: not x, bools))
    leaves_are = np.array([i.get_id() for i in nodelist])[bools]
    not_leaves = np.array([i.get_id() for i in nodelist])[opp_bools]

    # preorder == implicit that the clusters are all leaves! They're consistuent leaves...
    lst = [i.pre_order(lambda x: ind_to_id[x.id]) for i in np.array(nodelist)[opp_bools]] # getting list of all clusters 
    clstr_sizes = [len(i) for i in lst] # sizes of clusters
    cluster_bool = np.array(clstr_sizes) == clstrsz # only consider clusters of fixed size of the neuron numbers we want:
    dists = [i.dist for i in np.array(nodelist)[opp_bools]] # getting distances of all clusters 

    rel_clusters = np.array(lst, dtype=object)[cluster_bool] # all clusters containing clstrsz number. 
    rel_dists = np.array(dists, dtype=object)[cluster_bool] # all cluster distances containing clstrsz number.|
    rel_cluster_tuples = [tuple(i) for i in rel_clusters]

    rel_df = pd.DataFrame({'cluster':rel_cluster_tuples, f'dists':rel_dists})

    return rel_df.sort_values(by='dists', ascending=True)


def get_cluster_evolution(Z, targets, ind_to_id, max_era):
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
    condition=False
    i = 0
    inds = []
    j=0
    condition = [False] * max_era # get n top clusters
    while not all(condition):
        isit = len(s_set - set(rel_clusters[i])) == 0
        if isit:
            condition[j] = isit
            j+=1
            inds.append(i)
        i+=1
    i-=1
    return rel_clusters[inds] 

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



def entropy_calc(p):
    S = sum(-p * np.log(p))
    return S


def counting_types(nodes, type_dict):
    types = [type_dict[i] for i in nodes]
    count_dict = dict(Counter(types))
    return count_dict

def get_group_entropy(count_dict):
    '''
    Get Shannon entropy for a given set of nodes that are categorised according to type_dict

    '''
    count_types =list(count_dict.values())

    prob_vec = np.array(count_types)/ sum(count_types)
    entropy = entropy_calc(prob_vec)

    return entropy

def get_group_std(count_dict):
    '''
    Get standard deviation for a given set of nodes that are categorised according to type_dict

    '''
    count_types =list(count_dict.values())
    std = np.std(count_types)
    return std



def get_group_max(count_dict):
    '''
    Get max difference for a given set of nodes that are categorised according to type_dict

    '''
    count_types =list(count_dict.values())
    max_diff = max(count_types) - min(count_types)
    return max_diff

def get_cl_ev_ent(Z, targets, ind_to_id, type_dict, max_era):
    '''
    Z is the linkage matrix

    targets is the list of nodes we want present in our cluster

    ind_to_id is the list of nodes by which to index to give the node id. 

    max_era is the number of evolutionary eras of the cluster we wish to extract. 


    '''
    ntypes = len(set(type_dict.values()))

    max_ent = np.log(ntypes) # maximum entropy 

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
    condition=False
    i = 0
    inds = []
    j=0
    condition = [False] * max_era # get n top clusters
    entropies=[]
    residuals=[]
    while not all(condition):
        isit = len(s_set - set(rel_clusters[i])) == 0
        if isit:
            condition[j] = isit
            j+=1
            inds.append(i)

            nodes = rel_clusters[i]
            count_dict = counting_types(nodes=nodes, type_dict=type_dict)
            if len(count_dict)==ntypes:
                entropy = get_group_entropy(count_dict=count_dict)
                res = abs(entropy - max_ent)/max_ent
            else:
                entropy=None
                res=None
            entropies.append(entropy)
            residuals.append(res)

        i+=1
    i-=1


    return rel_clusters[inds], entropies, residuals

def above_thresh(threshold):
    def dummy(value):
        condition = value > threshold
        return condition
    return dummy

def below_thresh(threshold):
    def dummy(value):
        condition = value < threshold
        return condition
    return dummy


def gen_cl_func(Z, targets, ind_to_id, type_dict, test, func=get_group_entropy):
    '''
    Evaluates func using the generator function. 

    Z is the linkage matrix

    targets is the list of nodes we want present in our cluster

    ind_to_id is the list of nodes by which to index to give the node id. 


    '''
    ntypes = len(set(type_dict.values()))

    generate_next_cluster = generate_cluster_evolution(Z=Z, targets=targets, ind_to_id=ind_to_id)
    while True:
        cluster = next(generate_next_cluster)
        count_dict = counting_types(nodes=cluster, type_dict=type_dict)
        if len(count_dict)==ntypes:
            # print(count_dict)
            evaluation = func(count_dict=count_dict)
            condition = test(evaluation) | (len(cluster) > 100) 
            if condition:
                # print(cluster)
                return cluster, count_dict


    # return 


### 07_a_tree_distance.ipynb:

def perform_cluster_branching(Z, ind_to_id, targets, td):
    '''For a given linkage matrix Z according to ind_to_id which is the labels of ids used in Z,
    perform the cluster branching from each of the targets. 
    
    td is the dictionary of type of elements in the dendrogram. 

    return
        cluster_label_dict : a dict of {node_id : cluster_id}
        original_clusteirng : a dict of {cluster_id : list of corresponding node_ids }
        
    '''

    outputs = []
    for targ in targets:
        # o = gen_cl_func(Z=Z, targets=[targ], ind_to_id=ind_to_id, type_dict=td, test=above_thresh, func=get_group_entropy)
        
        # try:
        o = gen_cl_func(Z=Z, targets=[targ], ind_to_id=ind_to_id, type_dict=td, test=below_thresh(threshold=3), func=get_group_std)

        # vals= list(o[1].values())
        # res = abs(vals[0] - vals[1])

        res = get_group_std(o[1]) # using standard deviation to sort 

        outputs.append(list(o) + [res] + [len(o[0])])
        # except:
        #     continue
    sorted_clustering = sorted(outputs, key=lambda x: (x[2], -x[3]))  # ranked by standard deviation then largest first...  

    overall = set()
    save_clusters = []
    for i in sorted_clustering:
        if len(set(i[0]) & overall) < 1:
            save_clusters.append(i[0])
            overall |= set(i[0])


    list_of_cluster_labels = [dict(zip(save_clusters[i], [i for j in range(len(save_clusters[i]))])) for i in range(len(save_clusters))]

    cluster_label_dict = {}
    for d in list_of_cluster_labels:
        cluster_label_dict.update(d)
    # label_dict

    original_clustering = pd.DataFrame.from_dict(cluster_label_dict, orient='index', columns=['cluster']).reset_index().groupby('cluster')['index'].apply(list).to_dict()
    
    return cluster_label_dict, original_clustering


def dendrogram_clustering(Z, labels, clusters, label_dict=None, title_str='', t=0, save=False, axrange=None):

    '''Plot dendrogram given linkage Z, labels and clusters '''

    size = max([len(labels)*60/400, 5])
    plt.figure(figsize=(6,size))
    # only_clusters = set(np.hstack(clusters))
    if label_dict:
        # clstr_labels = [i + " : " + str(label_dict[i]) if i in only_clusters  else i for i in labels]
        clstr_labels = [i + " : " + str(label_dict.get(i, '')) for i in labels]

    else:
        clstr_labels=labels
        
    dendrograms = dendrogram(Z, labels=clstr_labels, get_leaves=True, orientation='left', color_threshold=t);

    colors = cm.rainbow(np.linspace(0,1,len(clusters)))
    ax = plt.gca()



    ylbls = ax.get_ymajorticklabels()
    for i in range(len(colors)):
        clstr = clusters[i]
        # print(clstr)
        for lbl in ylbls:
            label_colors = {'True' : colors[i], 'False' : 'black'}
            id_name = lbl.get_text().split(" : ")[0]
            is_clstr = id_name in clstr

            # print(lbl.get_text())
            # lbl.set_text(hemitype[5:])

            if is_clstr:
                lbl.set_color(label_colors[str(is_clstr)])


    plt.yticks(fontsize=10)
    plt.title(title_str)
    plt.xlim(axrange, 0)

    plt.tight_layout()
    if save:
        plt.savefig(f'{save}', bbox_inches='tight')
    plt.show()

def branch_growth(Z, ind_to_id, dist_mat, cluster_label_dict):


    ''' For given linkage matrix Z, and appropriate node ids ind_to_id used in Z, 
    and the cluster_label_dict which has already assigned some ids to a cluster label, 
    we assign the stray ids into already formed clusters by adding onto the closest preexisting cluster 
    in tree distance according to cluster_label_dict. 

    dist_mat is used to distinguish multiple clusters if there are many clusters which have the same 
    tree distance. 

    return 
        rel_cl_dict : a dictionary of {cluster_id : [node_id list]}
        expanded_cluster_dict : a dictionary of {node_id : cluster_id}
        
    '''
    tree_edges = get_network_from_tree(Z, ind_to_id)
    indices, adjlist = get_adj_from_edgelist(tree_edges)


    which_cluster = cluster_label_dict
    assigned = which_cluster.keys()

    to_group = set(ind_to_id) - set(assigned)

    dists, clustered, node_ids  = tree_type_search(nodes=to_group, targets=assigned, adjlist=adjlist, indices=indices, cl_dict=which_cluster)

    rehouse_strays = {}

    to_group = list(to_group)
    for i in range(len(to_group)):
        node_id = to_group[i]
        cl_ids = list(clustered[i]) # there may be more than one
        nid = node_ids[i]
        if len(cl_ids)>1:
            # print(node_id)
            closest_match = dist_mat.loc[[node_id], nid].T.sort_values(by=node_id, ascending=True).iloc[0].name
            cl_id = which_cluster[closest_match]
        else:
            cl_id = cl_ids[0]

        rehouse_strays[node_id] = cl_id

    expanded_cluster_dict = {}
    expanded_cluster_dict.update(which_cluster)
    expanded_cluster_dict.update(rehouse_strays)

    rev_cl_dict = pd.DataFrame.from_dict(expanded_cluster_dict, orient='index', columns=['cluster']).reset_index().groupby('cluster')['index'].apply(list).to_dict()
    expanded_clusters = list(rev_cl_dict.values())

    return rev_cl_dict, expanded_cluster_dict


def get_cluster_res(clusterings, type_dict, diff_func):
    '''
    Get the residuals of the clusters following function diff_func.
    '''
    df = pd.DataFrame({'clustering': clusterings})
    df['size'] = df.clustering.apply(len)

    type_series = df.clustering.apply(lambda x: counting_types(x, type_dict))
    count_series = type_series.apply(diff_func)
    df['types'] = type_series
    df['res'] = count_series

    return df
