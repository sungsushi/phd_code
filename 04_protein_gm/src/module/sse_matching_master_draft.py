import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from graspologic.match import graph_match
from matplotlib.colors import ListedColormap
import matplotlib as mpl

from .DME import DMatchEval

from .cmap_utils import prepare_adj_mat
from .nhood_vis import get_overlap_info_layout, invert_indices, compute_layouts_from_cont
from .cmap_runs import get_orientation_2, get_node_types, dict_to_sim_mat, dict_to_sim_mat_2, get_matching_full_inds, mapping_df

# from 15_gmatching_evalutate

def prepare_cm_adj_mat(fpath):
    '''prepares the distance matrix of the contact map'''
    data = pd.read_csv(fpath, delimiter = "\t")
    data = data.iloc[:, :-1]
    data.set_index('ss_ele', inplace=True)
    data.replace(to_replace='-', value=np.inf, inplace=True)
    data = data.apply(pd.to_numeric)

    return data

def get_contact_matrix(pid, dirpath='/Users/ssm47/Library/CloudStorage/OneDrive-UniversityofCambridge/ssnw'):
    '''Updated contact matrix getting from directory, and the full sse chain including coils.'''
    fpath = dirpath + '/'+ f'{pid}.ssnw'
    data = pd.read_csv(fpath, delimiter = "\t")
    # data = pd.read_csv(fpath)

    cmat = data.iloc[1:, 2:].dropna(axis=0, how='all').dropna(axis=1, how='all')

    lengths = data.Length[cmat.index.values].to_numpy()
    ele_inds = data.iloc[:,1][cmat.index.values].to_numpy()
    cmat.reset_index(drop=True, inplace=True)
    cmat.columns = cmat.index.values    
    new_indices = []
    for ele, length in zip(ele_inds, lengths):
        new_indices.append(ele[0] + str(int(length)))

    cmat.insert(loc=0, column='ss_ele', value=new_indices)
    cmat.replace(to_replace='-', value=np.inf, inplace=True)


    full_lengths = data.Length[1:].to_list()
    full_ssetypes = data.iloc[:,1][1:].to_list()
    full_sses = []

    for ele, length in zip(full_ssetypes, full_lengths):
        full_sses.append(ele[0] + str(int(length)))
    cmat.set_index('ss_ele', inplace=True)
    cmat = cmat.apply(pd.to_numeric)
    return cmat, full_sses

def get_orientation_multiadj(adj, orientation, types=False):
    ''' Nonbinary version: '''
    # adj = (adj * orientation).fillna(0).astype(int)

    if not types: 
        types = list(set(orientation.replace([np.inf], 0).astype(int).to_numpy().flatten()) - {0.})
    # print(types)

    multiadj = np.zeros((len(types), len(adj), len(adj)))
    # print(multiadj.shape)
    for t in range(len(types)):
        # print(types[t])
        t_mat = (orientation==types[t])*adj
        multiadj[t, :len(adj), :len(adj)] = t_mat

    return multiadj




def chain_adj(mat):
    '''Return an adjacency matrix where there are are edges between all consecutive elements.'''
    size = len(mat)
    chain = np.zeros((size, size))
    for i in range(size-1):
        # print(type(chain))
        chain[i, i+1] = 1
        chain[i+1, i] = 1
    return chain

def forward_chain_adj(mat):
    '''Return an adjacency matrix where there are are forward directed edges between consecutive elements.'''
    size = len(mat)
    chain = np.zeros((size, size))
    for i in range(size-1):
        chain[i, i+1] = 1
        # chain_adj[i+1, i] = 1
    return chain

def chain_adj_prep(val):
    '''Input the value of the similarity to create an adjacency matrix where there are edges between all consecutive elements.'''
    def _chain_adj(mat):
        size = len(mat)
        chain = np.zeros((size, size))
        for i in range(size-1):
            chain[i, i+1] = val
            chain[i+1, i] = val
        return chain
    return _chain_adj

def forward_chain_adj_prep(val):
    '''Input the value of the similarity to create an adjacency matrix where there are edges between all consecutive elements.'''
    def _chain_adj(mat):
        size = len(mat)
        chain = np.zeros((size, size))
        for i in range(size-1):
            chain[i, i+1] = val
            # chain_adj[i+1, i] = val
        return chain
    return _chain_adj


def forward_chain_w_adj(mat):
    '''Return the weighted adj matrix where there are edges between all consecutive elements'''
    binary = forward_chain_adj(mat)
    return binary * mat


def chain_w_adj(mat):
    '''Return the weighted adj matrix where there are edges between all consecutive elements'''
    binary = chain_adj(mat)
    return binary * mat

def get_cm_SS_order_orientation_graph(adj_df, col_name, orientation, chain_func=chain_adj):
    '''Get the contact map adjacency matrix with the orientation as
    the first three layers and last layer as the SS chain order edge type from adjacency df 
    also returns the structure
    Specifies the chain function chain_func 
     '''

    node_names = dict(zip(range(len(adj_df)), adj_df.index.values))
    adj_mat = adj_df.to_numpy()

    temp = nx.Graph()
    g = nx.from_numpy_array(adj_mat, create_using=temp)
    nx.set_node_attributes(g, node_names, name="CLASS")

    structure = pd.DataFrame({col_name:(node_names.values())}) 

    o_multiadj = get_orientation_multiadj(adj=adj_df, orientation=orientation)
    contact_o_multiadj = np.concatenate((o_multiadj, np.array([chain_func(adj_mat)])))

    return contact_o_multiadj, g, structure

def get_node_types(adj_df):
    '''Slicing the first element in the index names of the adjacency dataframe as the node type
    returns a dictionary between the numerical index and the node type.'''
    string = adj_df.index.values # node names 

    node_types = dict(zip(range(len(adj_df)), [i[0] for i in string]))
    return node_types

def prep_id_to_gmd_vars(fpath, o_fpath, sim_func,chain_func, col_name='id_1_structure'):
    adj = prepare_cm_adj_mat(fpath=fpath).replace(to_replace=np.inf, value=0)
    adj = sim_func(adj)
    o_1 = get_orientation_2(fpath=o_fpath)
    contact_o_multiadj, g, id_structure = get_cm_SS_order_orientation_graph(adj_df=adj, col_name=col_name, orientation=o_1, chain_func=chain_func)
    node_types = get_node_types(adj_df=adj)
    return contact_o_multiadj, g, node_types, id_structure

def exp_sim_func(adj):
    return abs(1/np.exp(adj/np.max(adj)))

def invd_sim_func(adj):
    return abs(1/(1+adj))


def exp_sim_func_norm(norm):
    def _exp_sim_func(adj):
        return abs(1/np.exp(adj/norm))
    return _exp_sim_func

def detect_community_pearson(adj, id_mapping, seed=0):
    '''Using an adjacency matrix, we detect communities in the network and index them using id mapping.
    graph one ids are indexed in numerical order of the adjacency matrix.
    graph two is the id_mapping indices corresponding to the second matrix. 
    '''
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    communities = nx.algorithms.community.louvain_communities(G, weight='weight', seed=seed)

    # print('\n')
    communities = list(map(list, communities))

    indexed_communities = [id_mapping[i] for i in communities]
    prsn_list = []
    for ind, val in enumerate(communities):
        prsn_list.append((np.vstack([val, indexed_communities[ind]]), pearsonr(x=val, y=indexed_communities[ind])))
        # print(val)
        # print(indexed_communities[ind])
        # print(pearsonr(x=val, y=indexed_communities[ind]))
        # print('\n')
    return prsn_list

def extract_doms(diff, sim_func, id_mapping, p=1, plot=True, save=False):
    sim_diff = sim_func(diff) 
    sim_diff_power = np.linalg.matrix_power(sim_diff, p)
    np.fill_diagonal(sim_diff_power, val=0)

    sim_diff_norm = sim_diff_power/sim_diff_power.max()

    if plot==True:
        plt.figure(figsize=(8,6))
        sns.heatmap(diff, annot=False, fmt=".1f")
        plt.title(r'$|d_A - Pd_BP^T|$')
        if save:
            plt.savefig(save, bbox_inches='tight', dpi=300)

        plt.show()

        plt.figure(figsize=(8,6))
        sns.heatmap(sim_diff_norm, annot=False, fmt=".1f")
        plt.title(rf'$s^{p}/max(s^{p})$')
        plt.show()
    # id_2_ind_to_id_1_ind = g2_to_g1_node_mapping.id_2_index.to_numpy()
    prsn_list = detect_community_pearson(adj=sim_diff_norm, id_mapping=id_mapping)
    return prsn_list


def co_domain_extract(val, id_1, id_2, sim_func, p_sim_func=None, p=1, plot=True, fpath=None, opath=None, use_lengths=False, alpha=1, gamma=1, save=False, pathext='.txt', fp=False):
    '''Corrected for alpha and gamma, magnitude and exponent of the length similarity term.'''

    if fpath==None:
        fpath = f'./data/100_sec_map/sscm/'
    fpath_1 = fpath + id_1 + pathext
    fpath_2 = fpath + id_2 + pathext

    if opath==None:
        opath = f'./data/100_sec_map/orientation/'
    o_fpath_1 = opath + id_1 + '.txt'
    o_fpath_2 = opath + id_2 + '.txt'

    # sim_func = exp_sim_func #exp_sim_func_norm(norm=norm)
    if not fp:
        rawadj_1 = prepare_cm_adj_mat(fpath=fpath_1)
        rawadj_2 = prepare_cm_adj_mat(fpath=fpath_2)
    if fp:
        rawadj_1, _ = get_contact_matrix(pid=id_1,dirpath=fpath)
        rawadj_2, _ = get_contact_matrix(pid=id_1,dirpath=fpath)

    rawadj_1.replace(to_replace=np.inf, value=0., inplace=True)
    rawadj_2.replace(to_replace=np.inf, value=0., inplace=True)

    norm = min([np.max(rawadj_2), np.max(rawadj_1)])
    if sim_func=='norm_invd_exp':
        sim_func = exp_sim_func_norm(norm=norm)

    returns = prep_id_to_gmd_vars(fpath=fpath_1, o_fpath=o_fpath_1, sim_func=sim_func, col_name='id_1_structure', chain_func=forward_chain_adj_prep(val=val))
    contact_o_multiadj_1, g_1, node_types_1, id_1_structure = returns
    returns = prep_id_to_gmd_vars(fpath=fpath_2, o_fpath=o_fpath_2, sim_func=sim_func, col_name='id_2_structure', chain_func=forward_chain_adj_prep(val=val))
    contact_o_multiadj_2, g_2, node_types_2, id_2_structure = returns

    graphs = {id_1:{'adj':contact_o_multiadj_1, 'ntypes':node_types_1, 'struct': id_1_structure, 'rawadj':rawadj_1}, id_2:{'adj':contact_o_multiadj_2, 'ntypes':node_types_2, 'struct':id_2_structure, 'rawadj':rawadj_2}}

    srtd = sorted([(graphs[k]['adj'], k) for k in [id_1, id_2]], key=lambda x:x[0].shape[1])
    id_A = srtd[0][1]
    id_B = srtd[1][1]
    # print(id_1, id_2)

    # print(id_A, id_B)

    g_A = graphs[id_A]['adj']
    g_B = graphs[id_B]['adj']
    # print(id_A, g_A.shape)
    # print(id_B, g_B.shape)
    raw_A = graphs[id_A]['rawadj']
    raw_B = graphs[id_B]['rawadj']

    node_types_A = graphs[id_A]['ntypes']
    node_types_B = graphs[id_B]['ntypes']
    # sim_mat = dict_to_sim_mat(node_types_1=node_types_A, node_types_2=node_types_B, sim_val=sim_val)
    sim_mat = dict_to_sim_mat_2(rawadj_1=raw_A, rawadj_2=raw_B, alpha=alpha, gamma=gamma, use_lengths=use_lengths)
    # print(sim_mat.shape)
    match_args={'rng':0, 'padding':'naive', 'transport':True, 'S':sim_mat, 'n_init':1}
    perm_inds_A, perm_inds_B, _, _ = graph_match(g_A, g_B, **match_args)
    all_inds = list(range(g_B.shape[1]))
    not_matched = [i for i in all_inds if i not in perm_inds_B]
    full_inds = np.concatenate([perm_inds_B, not_matched])

    g2_to_g1_node_mapping = mapping_df(structure_1=graphs[id_A]['struct'], structure_2=graphs[id_B]['struct'], colnames=['id_1_index', 'id_2_index'], full_inds=full_inds)
    # print(g2_to_g1_node_mapping.dropna())

    diff = abs(raw_B.to_numpy()[np.ix_(perm_inds_B, perm_inds_B)] - raw_A.to_numpy())

    larger_g_colname = g2_to_g1_node_mapping.dropna(axis=1).columns[-2]
    id_2_ind_to_id_1_ind = g2_to_g1_node_mapping[larger_g_colname].to_numpy()
    # prsn_list = detect_community_pearson(adj=sim_diff_norm, id_mapping=id_2_ind_to_id_1_ind)
    if p_sim_func==None: # if none then use the same one as before
        p_sim_func=sim_func

    prsn_list = extract_doms(diff=diff, sim_func=p_sim_func, id_mapping=id_2_ind_to_id_1_ind, p=p, plot=plot, save=save)
    return g2_to_g1_node_mapping, prsn_list


def full_diff_matrix_figure(diff, g2_to_g1_node_mapping,node_types_tex_2, node_types_tex_1, save=False, domain=False):
    '''Plot the full figure visualising the matched part of g2 and g1 and the difference matrix 
    that we get from the matching. 
    
    Element [i,j] is the element in matrix |AP - BP|. 

    Filled squares are for sheets and circles are for helices.
    '''

    # len_1 = g2_to_g1_node_mapping.id_1_structure.count()
    # len_2 = g2_to_g1_node_mapping.id_2_structure.count()

    colnames = g2_to_g1_node_mapping.columns
    index_names = [i for i in colnames if i[-5:]=='index']
    structure_names = [i.split('index')[0]+'structure' for i in index_names]
    sorted_index_ser = g2_to_g1_node_mapping[index_names].count().sort_values(ascending=True)
    id_index_1, id_index_2 = sorted_index_ser.index.to_numpy()
    len_1, len_2 = sorted_index_ser.values
    unmatched_n = len_2-len_1

    perm_inds = g2_to_g1_node_mapping.dropna().sort_values(id_index_1)[id_index_2].values
    mismatched = (~(g2_to_g1_node_mapping.dropna()[structure_names[0]].apply(lambda x: x[0]) == g2_to_g1_node_mapping.dropna()[structure_names[1]].apply(lambda x: x[0])))
    mm_inds = np.array(list(zip(np.arange(len_1), perm_inds)))[mismatched.values]

    nan_array = np.full((diff.shape[0], unmatched_n), np.nan)
    all_inds = list(range(len_2))
    full_diff_array = np.hstack((diff, nan_array))
    not_matched = [i for i in all_inds if i not in perm_inds]
    if len(not_matched)<1:
        full_inds=perm_inds
    else:
        full_inds = np.concatenate([perm_inds, not_matched])
    inv_inds = invert_indices(full_inds)

    data_toplot = full_diff_array[:,inv_inds] # the difference matrix transformed
    mask = np.isnan(data_toplot) # nan values in the difference matrix

    figsize = (len_2 * 0.375, len_1* 0.375)
    fig = plt.figure(figsize=figsize)


    cmap = sns.color_palette("viridis", as_cmap=True)  # Use a base colormap (e.g., 'viridis')
    cmap = cmap(np.arange(cmap.N))  # Convert to an array
    cmap[:,-1] = 1  # Set the alpha to fully opaque
    cmap = ListedColormap(cmap)  # Convert back to colormap

    gs0 = mpl.gridspec.GridSpec(2,2, figure=fig,
                                    height_ratios=[10,1],width_ratios=[50,1],  hspace=0.05, wspace=0)


    # Create the colorbar axis
    cbar_ax = fig.add_subplot(gs0[0, 1])

    # Create the heatmap axis
    heatmap_ax = fig.add_subplot(gs0[0, 0])

    # plot heatmap:
    sns.heatmap(data_toplot, mask=mask, cbar=True, linewidths=0., cbar_ax=cbar_ax,\
                linecolor=None, vmin=np.nanmin(data_toplot), vmax=np.nanmax(data_toplot), ax=heatmap_ax)

    # x,y bounds for the heatmap
    xmin,xmax = heatmap_ax.get_xlim()
    ymin,ymax = heatmap_ax.get_ylim()

    # grey out the nans in the figure
    heatmap_ax.imshow(mask, cmap=ListedColormap(['#e3e3e3']), interpolation='none', alpha=1, zorder=0, aspect='equal', 
            extent =[xmin,xmax,ymin, ymax])

    heatmap_ax.set_title(r'$|PA - PB|$')
    xlabels = [node_types_tex_2[item.get_text()] for item in heatmap_ax.get_xticklabels()]

    heatmap_ax.xaxis.tick_top()

    heatmap_ax.set_xticklabels(xlabels)

    ylabels = [node_types_tex_1[item.get_text()] for item in heatmap_ax.get_yticklabels()]

    heatmap_ax.set_yticklabels(ylabels)

    for y,x in mm_inds:
        heatmap_ax.get_xticklabels()[x].set_color('red')
        heatmap_ax.get_yticklabels()[y].set_color('red')

    filled_marker_style = dict(marker='s', linestyle=':', markersize=16,mew=2, 
                            color='tab:blue',
                            markerfacecolor='tab:blue',
                            markerfacecoloralt='lightsteelblue',
                            markeredgecolor='tab:blue')
    
    mm_marker_style = dict(marker='x',linestyle='', markersize=16, mew=2, 
                        color='tab:red',
                        # markerfacecolor='tab:blue',
                        # markerfacecoloralt='lightsteelblue',
                        markeredgecolor='white')

    matches = np.array(list(zip(np.arange(len_1), perm_inds))) + 0.5

    # Plot the mapping from the graph matching:
    heatmap_ax.plot(matches[:,1], matches[:,0], fillstyle='none', **filled_marker_style)
    heatmap_ax.plot(mm_inds[:,1] + 0.5, mm_inds[:,0] + 0.5, fillstyle='none', **mm_marker_style)

    if domain:
        for ith, d in enumerate(domain):
            domain_nodes_1 = domain[ith][0][0]
            # print(domain_nodes_1)
            heatmap_ax.plot(matches[:,1][domain_nodes_1], matches[:,0][domain_nodes_1], fillstyle='none', marker=fr'${ith}$', linestyle='', color='w')

    heatmap_ax.spines['top'].set_visible(True)
    heatmap_ax.spines['right'].set_visible(True)
    heatmap_ax.spines['bottom'].set_visible(True)
    heatmap_ax.spines['left'].set_visible(True)

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
    plt.show()


def get_graphcont(id_1, id_2, th, fpath=None, opath=None):
    if fpath==None:
        fpath = f'./data/100_sec_map/sscm/'
    fpath_1 = fpath + id_1 + '.txt'
    fpath_2 = fpath + id_2 + '.txt'

    if opath==None:
        opath = f'./data/100_sec_map/orientation/'
    o_fpath_1 = opath + id_1 + '.txt'
    o_fpath_2 = opath + id_2 + '.txt'

    # fpath = f'./data/sec_network/sscm/{id_1}.txt'
    # adj_1 = prepare_cm_adj_mat(fpath=fpath).replace(to_replace=np.inf, value=0.)
    # print("Threshold for edges:", th)
    adj_1 = prepare_adj_mat(fpath=fpath_1, threshold=th)


    adj_mat_1 = adj_1.to_numpy()
    node_names_1 = dict(zip(range(len(adj_1)), adj_1.index.values))
    temp = nx.Graph()
    g_1 = nx.from_numpy_array(adj_mat_1, create_using=temp)
    nx.set_node_attributes(g_1, node_names_1, name="CLASS")

    # o_fpath = f'./data/sec_network/orientation/{id_1}.txt'
    o_1 = get_orientation_2(fpath=o_fpath_1)
    contact_o_multiadj_1, g_1, id_1_structure = get_cm_SS_order_orientation_graph(adj_df=adj_1, col_name='id_1_structure', orientation=o_1, chain_func=forward_chain_adj)
    node_types_1 = get_node_types(adj_df=adj_1)


    # fpath = f'./data/sec_network/sscm/{id_2}.txt'
    # adj_2 = prepare_cm_adj_mat(fpath=fpath).replace(to_replace=np.inf, value=0.)
    adj_2 = prepare_adj_mat(fpath=fpath_2, threshold=th)

    adj_mat_2 = adj_2.to_numpy()
    node_names_2 = dict(zip(range(len(adj_2)), adj_2.index.values))
    temp = nx.Graph()
    g_2 = nx.from_numpy_array(adj_mat_2, create_using=temp)
    nx.set_node_attributes(g_2, node_names_2, name="CLASS")
    # o_fpath = f'./data/sec_network/orientation/{id_2}.txt'
    o_2 = get_orientation_2(fpath=o_fpath_2)
    contact_o_multiadj_2, g_2, id_2_structure = get_cm_SS_order_orientation_graph(adj_df=adj_2, col_name='id_2_structure', orientation=o_2, chain_func=forward_chain_adj)
    node_types_2 = get_node_types(adj_df=adj_2)
    # print(len(adj_2))

    graphs = [[ g_1, contact_o_multiadj_1],[g_2, contact_o_multiadj_2 ]]
    return graphs, node_types_1, node_types_2


def draw_overlap_from_outputs(output,node_types_1, node_types_2, domain=False, save=False):
    '''From the output of compute_layouts_from_cont function and the domain assignment, visualise the matching of protein SS.'''
    sdict = {'E':'s', 'H':'o'}
    overlap_graphs = get_overlap_info_layout(output, directed=False)

    g1_E = [key for key, val in node_types_1.items() if val=='E']
    g1_H = [key for key, val in node_types_1.items() if val=='H']

    g2_E = [key for key, val in node_types_2.items() if val=='E']
    g2_H = [key for key, val in node_types_2.items() if val=='H']

    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_nodes(
        output['graph_2'], nodelist=g2_E, pos=output['layout_2'],
        node_color='#dceadc', edgecolors='#a3c8a3', alpha=1, node_shape=sdict['E'],
        label='g2 sheet', ax=ax
    )
    nx.draw_networkx_nodes(
        output['graph_2'], nodelist=g2_H, pos=output['layout_2'],
        node_color='#dceadc', edgecolors='#a3c8a3', alpha=1, node_shape=sdict['H'],
        label='g2 helix', ax=ax
    )
    nx.draw_networkx_edges(
        output['graph_2'], pos=output['layout_2'], alpha=1, style='--',
        edge_color='#a3c8a3', label='g2 edges', ax=ax
    )

    nx.draw_networkx_nodes(
        output['graph_1'], nodelist=g1_E, pos=output['layout_1'],
        node_color='None', linewidths=2, alpha=1, edgecolors='r',
        node_shape=sdict['E'], label='g1 sheet', ax=ax
    )
    nx.draw_networkx_nodes(
        output['graph_1'], nodelist=g1_H, pos=output['layout_1'],
        node_color='None', linewidths=2, alpha=1, edgecolors='r',
        node_shape=sdict['H'], label='g1 helix', ax=ax
    )
    nx.draw_networkx_edges(
        output['graph_1'], pos=output['layout_1'], alpha=1,
        edge_color='#ffb8b8', width=2, label='g1 edges', ax=ax
    )

    nx.draw_networkx_edges(
        overlap_graphs['o_g'], pos=output['layout_1'],
        label='common edges', width=2, ax=ax
    )

    if domain:
        for ith, d in enumerate(domain):
            domain_nodes_1 = domain[ith][0][0]
            nx.draw_networkx_nodes(
                output['graph_1'], nodelist=domain_nodes_1, pos=output['layout_1'],
                alpha=1, node_shape=fr'${ith}$', node_size=50, label=f'{ith}th domain', ax=ax
            )

    ax.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)

    if not save:
        plt.show()

    return fig


    # the following code is a solid color version...
    # plt.figure(figsize=(10,10))
    # # plt.title(f'Threshold={th}')

    # nx.draw_networkx_nodes(output['graph_2'], nodelist=g2_E, pos=output['layout_2'], node_color='#dceadc',edgecolors='#a3c8a3', alpha=1, node_shape=sdict['E'], label='g2 sheet')
    # nx.draw_networkx_nodes(output['graph_2'],nodelist=g2_H, pos=output['layout_2'], node_color='#dceadc',edgecolors='#a3c8a3', alpha=1, node_shape=sdict['H'], label='g2 helix')
    # nx.draw_networkx_edges(output['graph_2'],pos=output['layout_2'], alpha=1, style='--', edge_color='#a3c8a3', label='g2 edges')

    # nx.draw_networkx_nodes(output['graph_1'], nodelist=g1_E, pos=output['layout_1'], node_color='None',linewidths=2,  alpha=1, edgecolors='r', node_shape=sdict['E'], label='g1 sheet')
    # nx.draw_networkx_nodes(output['graph_1'],nodelist=g1_H, pos=output['layout_1'], node_color='None',linewidths=2,  alpha=1, edgecolors='r', node_shape=sdict['H'], label='g1 helix')
    # nx.draw_networkx_edges(output['graph_1'],pos=output['layout_1'], alpha=1, edge_color='#ffb8b8', width=2, label='g1 edges')

    # nx.draw_networkx_edges(overlap_graphs['o_g'], pos=output['layout_1'], label='common edges', width=2, )

    # # dcmap = mpl.colormaps['Dark2'].colors
    # if domain:
    #     for ith, d in enumerate(domain):
    #         domain_nodes_1 = domain[ith][0][0]
    #         nx.draw_networkx_nodes(output['graph_1'], nodelist=domain_nodes_1, pos=output['layout_1'], alpha=1, node_shape=fr'${ith}$', node_size=50, label=f'{ith}th domain')
    # plt.legend(fontsize=15)
    # plt.tight_layout()
    # if save:
    #     plt.savefig(save, bbox_inches='tight', dpi=300)
        
    # plt.show()


def dist_diff(id_1, id_2, perm_inds, fpath=None):
    if fpath==None:
        fpath = f'./data/100_sec_map/sscm/'
    fpath_1 = fpath + id_1 + '.txt'
    fpath_2 = fpath + id_2 + '.txt'
    adj_1 = prepare_cm_adj_mat(fpath=fpath_1).replace(to_replace=np.inf, value=0.)
    adj_2 = prepare_cm_adj_mat(fpath=fpath_2).replace(to_replace=np.inf, value=0.)
    diff = abs(adj_2.to_numpy()[np.ix_(perm_inds, perm_inds)] - adj_1.to_numpy())
    return diff