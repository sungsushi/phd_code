import numpy as np 
import pandas as pd 
import networkx as nx



# from celegans/09_network_metrics.ipynb

def get_shell_graph(g):
    '''For a given networkx directed graph g, we make the undirected shell graph, set the attributes for weight and inverse weight.'''
    g_undirected = nx.from_numpy_array(nx.adjacency_matrix(g).todense().T + nx.adjacency_matrix(g).todense(), create_using=nx.Graph) 
    # The directed graph plus the transpose of it to create the shell graph

    g_undirected = nx.relabel_nodes(g_undirected, mapping=dict(enumerate(g.nodes())))
    labels = nx.get_edge_attributes(g_undirected,'weight')
    inv_weight_labels = {key:1/labels[key] for key in labels}
    nx.set_node_attributes(g_undirected, values= inv_weight_labels, name='inv_weight')
    return g_undirected


def get_centrality_df(edges, weighted=False): # aka get_centrality_2_df
    '''from a df of edges, get the closeness, betweenness and eigenvector centrality'''

    g = nx.DiGraph()
    
    if weighted:
        slice_string = ['pre', 'post', 'attr']
        w_str = 'weight'
        iw_str = 'inv_weight'
    if not weighted:
        slice_string = ['pre', 'post']
        w_str = None
        iw_str = None


    g.add_edges_from(edges[slice_string].to_numpy())
    
    undirected_g = get_shell_graph(g)
    in_harmonic = nx.harmonic_centrality(g, distance=iw_str)
    out_harmonic = nx.harmonic_centrality(g.reverse(), distance=iw_str)
    out_deg_cent = nx.out_degree_centrality(g)
    in_deg_cent = nx.in_degree_centrality(g)
    deg_cent = nx.degree_centrality(g)
    # bcent = nx.betweenness_centrality(g, weight=iw_str)
    # u_bcent = nx.betweenness_centrality(undirected_g, weight=iw_str)
    # cf_bcent = nx.current_flow_betweenness_centrality(g, weight=iw_str)
    u_cf_bcent = nx.current_flow_betweenness_centrality(undirected_g, weight=iw_str)
    in_pagerank = nx.pagerank(g, weight=w_str)
    out_pagerank = nx.pagerank(g.reverse(), weight=w_str)
    # out_ecent = nx.eigenvector_centrality(g.reverse(), weight=w_str, max_iter=200) # to find eigenvectors of matrix for out-edges
    # in_ecent = nx.eigenvector_centrality(g, weight=w_str, max_iter=200) # to find eigenvectors of matrix for out-edges

    centralities = {'in_harmonic':in_harmonic,'out_harmonic':out_harmonic,\
                    'in_pagerank':in_pagerank, 'out_pagerank':out_pagerank,\
                    'degree':deg_cent, 'in_degree':in_deg_cent, 'out_degree':out_deg_cent,\
                    'u_cf_bcent':u_cf_bcent}
    if weighted:
        strength = dict(nx.degree(g, weight='weight'))
        in_strength = dict(g.in_degree(weight='weight'))
        out_strength = dict(g.out_degree(weight='weight'))
        centralities['strength'] = strength
        centralities['in_strength'] = in_strength
        centralities['out_strength'] = out_strength
    c_df = pd.DataFrame(centralities)
    return c_df